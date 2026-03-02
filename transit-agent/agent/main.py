"""Async entrypoint: vehicle API, display WS, VAD/STT, proactive loop, LLM turn on each transcript."""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic
import config
from agent.audio_input import transcript_generator
from agent.audio_output import play_local_file, speak
from agent.context import RideContext, make_mock_context
from agent import display_server
from agent import echo_guard
from agent.llm import run_turn, add_proactive_offer
from agent.proactive import proactive_loop
from agent import spotify_token_server

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# Startup intros by rider type (fixed copy, no LLM)
INTRO_COMMUTER = (
    "Welcome aboard. This is Gladys — I can adjust the lights, temperature, and music, "
    "or answer questions about your route. Just speak whenever you're ready."
)
INTRO_DEMO = (
    "Welcome aboard a Glydways vehicle — transit, designed for riders. I'm Gladys, your in-cabin "
    "assistant for this ride. I can control the lights, temperature, and audio, keep you updated on "
    "your route, or just answer questions. You don't need to memorize commands — if there's something "
    "I can help with, I'll let you know. Otherwise, just speak naturally and I'm here."
)


def get_ride_context() -> RideContext:
    """Provide current ride context (for proactive loop and LLM). Replace with real source."""
    return make_mock_context(
        elapsed_seconds=get_ride_context._elapsed,
        eta_seconds=max(0, 180 - (get_ride_context._elapsed // 10)),
    )


get_ride_context._elapsed = 0

# Shared state for proactive offers and conversation (proactive uses its own conversation list)
offers_made_shared: list[str] = []
proactive_conversation: list[dict] = []

# Only one LLM turn + TTS at a time (user or proactive) to avoid overlapping responses
_turn_lock = asyncio.Lock()

# Proactive announcements run only after this many seconds of silence (no user or proactive turn)
PROACTIVE_MIN_SILENCE_SEC = 45
_last_turn_end_time: float = 0.0

# Dedupe transcripts: same/similar phrase within this window is ignored (VAD/Whisper sometimes double-emits)
_TRANSCRIPT_DEDUPE_SEC = 8
_last_transcript_key: str = ""
_last_transcript_time: float = 0


def _transcript_seen_recently(transcript: str) -> bool:
    """True if we already processed this (normalized) transcript in the last _TRANSCRIPT_DEDUPE_SEC."""
    import time
    key = transcript.strip().lower() or ""
    if not key:
        return True
    now = time.monotonic()
    if key == _last_transcript_key and (now - _last_transcript_time) < _TRANSCRIPT_DEDUPE_SEC:
        return True
    return False


def _mark_transcript_processed(transcript: str) -> None:
    import time
    global _last_transcript_key, _last_transcript_time
    _last_transcript_key = (transcript.strip().lower() or "")
    _last_transcript_time = time.monotonic()


async def _speak_immediate_ack(ack_text: str) -> None:
    """Register ack with echo guard, update display, and speak so user knows they were heard."""
    echo_guard.register_utterance(ack_text)
    await display_server.send_layout("speaking", {"text": ack_text})
    await speak(ack_text)


async def on_proactive_trigger(trigger_key: str, user_message: str) -> None:
    """Called when a proactive trigger fires: inject message and get LLM to respond."""
    global _last_turn_end_time
    async with _turn_lock:
        ctx = get_ride_context()
        add_proactive_offer(offers_made_shared, trigger_key)
        text, _ = await run_turn(
            user_message, ctx, offers_made_shared, proactive_conversation,
            on_immediate_ack=_speak_immediate_ack,
        )
        if text:
            await display_server.send_layout("speaking", {"text": text})
            await speak(text)
        await display_server.send_layout("idle", ctx.cabin.to_dict())
        _last_turn_end_time = time.monotonic()


async def main() -> None:
    global _last_turn_end_time
    if not (config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY.strip()):
        sys.exit(
            "ANTHROPIC_API_KEY is not set. Add it to transit-agent/.env or set the environment variable.\n"
            "Get a key at https://console.anthropic.com/"
        )
    # 1. Start mock vehicle API in subprocess
    import subprocess
    transit_agent_root = Path(__file__).resolve().parent.parent
    proc = subprocess.Popen(
        [sys.executable, "-m", "vehicle_api.server"],
        cwd=str(transit_agent_root),
        env={
            **__import__("os").environ,
            "VEHICLE_API_PORT": str(config.VEHICLE_API_PORT),
            "PYTHONPATH": str(transit_agent_root),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait for API to be up
        import httpx
        ws_task = None
        proactive_task = None
        spotify_token_task = None
        for _ in range(50):
            try:
                r = httpx.get(f"http://127.0.0.1:{config.VEHICLE_API_PORT}/state", timeout=1.0)
                r.raise_for_status()
                break
            except Exception:
                await asyncio.sleep(0.2)
        else:
            raise RuntimeError("Vehicle API did not start")

        # 2. Start WebSocket server for display (background task)
        ws_task = asyncio.create_task(display_server.run(config.WS_PORT))

        # 2b. Spotify token server (for display Web Playback SDK) when Spotify is configured
        spotify_token_task = None
        if config.USE_SPOTIFY:
            spotify_token_task = asyncio.create_task(spotify_token_server.run(config.SPOTIFY_TOKEN_PORT))

        # 3. Proactive loop: only runs after PROACTIVE_MIN_SILENCE_SEC of no user/proactive turns
        offered: set[str] = set()

        def _proactive_can_run() -> bool:
            return (time.monotonic() - _last_turn_end_time) >= PROACTIVE_MIN_SILENCE_SEC

        proactive_task = asyncio.create_task(
            proactive_loop(
                lambda: get_ride_context(),
                on_proactive_trigger,
                offered,
                can_run_now=_proactive_can_run,
            )
        )

        # 4. Seed display state immediately so late-joining clients get ride data on connect; then wait for client and re-send
        ctx0 = get_ride_context()
        await display_server.send_layout("idle", {
            "route_name": ctx0.route_name,
            "next_stop": ctx0.next_stop,
            "eta_seconds": ctx0.eta_seconds,
            "progress_pct": 0,
        })
        await display_server.wait_for_client(timeout=2.0)
        # Re-send so any client that connected during wait gets ride content before intro
        await display_server.send_layout("idle", {
            "route_name": ctx0.route_name,
            "next_stop": ctx0.next_stop,
            "eta_seconds": ctx0.eta_seconds,
            "progress_pct": 0,
        })

        # 4.5. Play alert sound then introduce itself once at startup (fixed copy by rider type)
        intro_ctx = get_ride_context()
        intro_text = INTRO_DEMO if config.RIDER_TYPE == "demo" else INTRO_COMMUTER
        alert_path = Path(__file__).resolve().parent.parent / "assets" / "voicePromptAlert2.mp3"
        if alert_path.is_file():
            async with _turn_lock:
                await play_local_file(alert_path)
        echo_guard.register_utterance(intro_text)  # so delayed echo of intro is not treated as user input
        async with _turn_lock:
            await display_server.send_layout("speaking", {"text": intro_text})
            await speak(intro_text)
            _last_turn_end_time = time.monotonic()
        offered.add("boarding")
        await display_server.send_layout("idle", {
            "route_name": intro_ctx.route_name,
            "next_stop": intro_ctx.next_stop,
            "eta_seconds": intro_ctx.eta_seconds,
            "progress_pct": 0,
        })

        # 5. VAD + STT → LLM → TTS (echo guarding via echo_guard in audio_input + audio_output)
        echo_guard.clear()
        conversation: list[dict] = []
        offers_made_shared.clear()

        async for transcript, transcript_ts in transcript_generator(
            model_name=config.WHISPER_MODEL,
            input_device=config.AUDIO_INPUT_DEVICE,
        ):
            if _transcript_seen_recently(transcript):
                logger.debug("Skipping duplicate transcript: %r", transcript[:50])
                continue
            _mark_transcript_processed(transcript)
            logger.info("User said: %s", transcript)
            get_ride_context._elapsed += 60  # Simulate time passing; replace with real clock

            ctx = get_ride_context()
            try:
                r = httpx.get(f"http://127.0.0.1:{config.VEHICLE_API_PORT}/state", timeout=2.0)
                if r.is_success:
                    from vehicle_api.state import CabinState
                    ctx.cabin = CabinState.from_dict(r.json())
            except Exception:
                pass

            async with _turn_lock:
                try:
                    text, conversation = await run_turn(
                        transcript, ctx, offers_made_shared, conversation,
                        on_immediate_ack=_speak_immediate_ack,
                    )
                    if text:
                        logger.info("Speaking: %s", text[:80] + "..." if len(text) > 80 else text)
                        await display_server.send_layout("speaking", {"text": text})
                        await speak(text)
                    else:
                        logger.warning("No response text to speak")
                except anthropic.BadRequestError as e:
                    err_msg = str(e)
                    try:
                        body = getattr(e, "body", None) or getattr(e, "response", None)
                        if isinstance(body, dict) and "error" in body:
                            err_msg = body["error"].get("message", err_msg)
                    except Exception:
                        pass
                    logger.error("Anthropic API error (e.g. low credits): %s", err_msg)
                    fallback = "I'm having trouble reaching my brain right now. Please try again in a moment."
                    await display_server.send_layout("speaking", {"text": fallback})
                    await speak(fallback)
                except anthropic.APIError as e:
                    logger.exception("Anthropic API error")
                    fallback = "Something went wrong on my end. Please try again."
                    await display_server.send_layout("speaking", {"text": fallback})
                    await speak(fallback)
                await display_server.send_layout("idle", {
                    "route_name": ctx.route_name,
                    "next_stop": ctx.next_stop,
                    "eta_seconds": ctx.eta_seconds,
                    "progress_pct": min(90, get_ride_context._elapsed * 100 // max(1, ctx.ride_duration_seconds)),
                })
                _last_turn_end_time = time.monotonic()

    finally:
        if proactive_task is not None:
            proactive_task.cancel()
        if spotify_token_task is not None:
            spotify_token_task.cancel()
        if ws_task is not None:
            ws_task.cancel()
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())
