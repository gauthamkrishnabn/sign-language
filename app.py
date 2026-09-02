"""
app.py
------
Streamlit app: upload a video of someone signing, get back a text
transcript. Also lets you build up the "vocabulary" the app recognizes
by uploading short labeled example clips (one sign per clip).

Run locally:    streamlit run app.py
Deploy:         see README.md
"""

import os
import tempfile

import streamlit as st

from landmarks import extract_landmarks_from_video
from segmentation import segment_signs
import templates as tpl
from recognizer import recognize_video
from sentence import rule_based_sentence, polish_with_claude
import batch_add

TEMPLATE_DIR = os.environ.get("SIGN_TEMPLATE_DIR", "templates")

st.set_page_config(page_title="Sign Language → Text", page_icon="🤟", layout="wide")


def save_upload_to_tempfile(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def render_sidebar():
    st.sidebar.header("Settings")
    sample_fps = st.sidebar.slider("Sampling rate (fps)", 5, 30, 15,
                                    help="Higher = more accurate, slower to process.")
    motion_threshold = st.sidebar.slider("Motion threshold", 0.005, 0.10, 0.02, 0.005,
                                          help="Lower = more sensitive to small movements "
                                               "(may over-segment). Higher = only catches "
                                               "big, clear movements.")
    st.sidebar.divider()
    st.sidebar.subheader("Optional: fluent-sentence polish")
    use_claude = st.sidebar.checkbox("Smooth transcript with Claude", value=False)
    api_key = None
    if use_claude:
        api_key = st.sidebar.text_input("Anthropic API key", type="password",
                                         help="Only used to turn the raw recognized "
                                              "words into a natural sentence. Not stored.")
    st.sidebar.divider()
    vocab = tpl.vocab_size(TEMPLATE_DIR)
    st.sidebar.caption(f"Vocabulary size: **{vocab} word(s)**")
    if vocab == 0:
        st.sidebar.warning("No signs in your library yet — add some in the "
                            "'Build Vocabulary' tab before translating a video.")
    return sample_fps, motion_threshold, use_claude, api_key


def render_translate_tab(sample_fps, motion_threshold, use_claude, api_key):
    st.subheader("Upload a video to translate")
    video_file = st.file_uploader("Video file (mp4, mov, avi...)",
                                   type=["mp4", "mov", "avi", "mkv", "webm"],
                                   key="translate_upload")

    if video_file is None:
        st.info("Upload a video of someone signing, clearly and at a "
                "moderate pace, with brief pauses between words.")
        return

    st.video(video_file)

    if st.button("Translate", type="primary"):
        templates_lib = tpl.load_templates(TEMPLATE_DIR)
        if not templates_lib:
            st.error("Your vocabulary library is empty. Add example signs "
                      "in the 'Build Vocabulary' tab first.")
            return

        video_path = save_upload_to_tempfile(video_file)
        try:
            with st.spinner("Extracting hand & body landmarks..."):
                frames, timestamps = extract_landmarks_from_video(
                    video_path, sample_fps=sample_fps)

            if len(frames) == 0:
                st.error("Couldn't read any frames from that video.")
                return

            with st.spinner("Finding sign boundaries..."):
                segments = segment_signs(frames, timestamps,
                                          motion_threshold=motion_threshold)

            if not segments:
                st.warning("No distinct signing motion detected. Try lowering "
                            "the motion threshold in the sidebar.")
                return

            with st.spinner(f"Matching {len(segments)} segment(s) against your "
                             f"{len(templates_lib)}-word vocabulary..."):
                results = recognize_video(frames, segments, templates_lib)

            gloss_sequence = [r["gloss"] for r in results if r["gloss"]]

            st.markdown("### Recognized signs")
            for r in results:
                label = r["gloss"] or "— (no match)"
                flag = "" if r["confident"] else "  ⚠️ low confidence"
                st.write(f"`{r['start_time']:.1f}s–{r['end_time']:.1f}s`  →  "
                         f"**{label}**{flag}")

            st.markdown("### Transcript")
            base_sentence = rule_based_sentence(gloss_sequence)
            final_sentence = base_sentence

            if use_claude and api_key:
                with st.spinner("Polishing into a fluent sentence..."):
                    polished = polish_with_claude(gloss_sequence, api_key=api_key)
                if polished:
                    final_sentence = polished
                else:
                    st.caption("Couldn't reach Claude — showing raw transcript instead.")

            st.success(final_sentence)
            st.download_button("Download transcript (.txt)", final_sentence,
                                file_name="transcript.txt")
        finally:
            os.unlink(video_path)


def render_vocabulary_tab(sample_fps):
    st.subheader("Build your sign vocabulary")
    st.caption(
        "This app recognizes signs by comparing new video against labeled "
        "examples you provide — there's no giant pretrained sign-language "
        "model behind it. Record a short clip of *one word/sign* at a time "
        "(a few seconds, hands returning to rest before/after), label it, "
        "and add it below. 2–3 examples per word, from slightly different "
        "angles/speeds, noticeably improves accuracy."
    )

    manifest = tpl.load_manifest(TEMPLATE_DIR)
    if manifest:
        st.write("**Current vocabulary:**")
        cols = st.columns(4)
        for i, (slug, files) in enumerate(sorted(manifest.items())):
            with cols[i % 4]:
                st.write(f"• {slug} ({len(files)} example{'s' if len(files) != 1 else ''})")
        remove_word = st.selectbox("Remove a word", ["—"] + sorted(manifest.keys()))
        if remove_word != "—" and st.button("Remove"):
            tpl.remove_word(TEMPLATE_DIR, remove_word)
            st.rerun()
        st.divider()

    with st.form("add_template_form", clear_on_submit=True):
        gloss = st.text_input("Word / sign label (e.g. 'hello', 'thank you')")
        clip = st.file_uploader("Short video clip of just this sign", type=[
            "mp4", "mov", "avi", "mkv", "webm"])
        submitted = st.form_submit_button("Add to vocabulary")

    if submitted:
        if not gloss or not clip:
            st.error("Please provide both a label and a video clip.")
        else:
            clip_path = save_upload_to_tempfile(clip)
            try:
                with st.spinner("Extracting landmarks..."):
                    frames, _ = extract_landmarks_from_video(clip_path, sample_fps=sample_fps)
                if len(frames) == 0:
                    st.error("Couldn't read any frames from that clip.")
                else:
                    tpl.add_template(TEMPLATE_DIR, gloss, frames)
                    st.success(f"Added example for '{gloss}'.")
                    st.rerun()
            finally:
                os.unlink(clip_path)


def render_batch_add_tab(sample_fps, motion_threshold):
    st.subheader("Batch-add from one video")
    st.caption(
        "Faster than uploading one clip per word: record yourself signing a "
        "*list of words in order*, with a clear pause between each one, then "
        "type that same list here (same order). The app auto-detects each "
        "sign's boundaries and saves them all as templates in one go."
    )

    clip = st.file_uploader(
        "Video with multiple signs, performed in sequence with pauses between them",
        type=["mp4", "mov", "avi", "mkv", "webm"], key="batch_upload")
    words_text = st.text_area(
        "Words, one per line, IN THE ORDER they're signed",
        placeholder="hello\nthank you\nwater\nplease",
        height=150,
    )
    words = [w.strip() for w in words_text.splitlines() if w.strip()]

    col1, col2 = st.columns(2)
    local_threshold = col1.slider("Motion threshold (this tab only)", 0.005, 0.10,
                                   motion_threshold, 0.005, key="batch_threshold",
                                   help="Adjust and re-preview if the detected segment "
                                        "count doesn't match your word count.")
    min_gap = col2.slider("Pause length required between signs (frames)", 1, 15, 3,
                           key="batch_min_gap")

    if clip is not None and st.button("Preview segments", type="primary"):
        clip_path = save_upload_to_tempfile(clip)
        try:
            with st.spinner("Extracting landmarks and detecting sign boundaries..."):
                frames, segments = batch_add.preview_segments(
                    clip_path, sample_fps=sample_fps,
                    motion_threshold=local_threshold, min_gap_frames=min_gap)
            st.session_state["batch_frames"] = frames
            st.session_state["batch_segments"] = segments
        finally:
            os.unlink(clip_path)

    segments = st.session_state.get("batch_segments")
    frames = st.session_state.get("batch_frames")

    if segments is not None:
        st.markdown(f"### Detected {len(segments)} segment(s) — you listed {len(words)} word(s)")
        n = max(len(segments), len(words))
        for i in range(n):
            seg_label = (f"`{segments[i][2]:.1f}s–{segments[i][3]:.1f}s`"
                         if i < len(segments) else "*(no segment detected)*")
            word_label = words[i] if i < len(words) else "*(no word given)*"
            st.write(f"{i+1}. {seg_label}  →  **{word_label}**")

        if len(segments) != len(words):
            st.warning(
                "Counts don't match, so nothing can be saved yet. Try lowering the "
                "motion threshold if signs got merged together, or raising it if "
                "one sign got split into two — then preview again."
            )
        else:
            if st.button("Save all as templates", type="primary"):
                saved = batch_add.commit_segments(TEMPLATE_DIR, frames, segments, words)
                st.success(f"Saved {len(saved)} template(s): " +
                           ", ".join(w for w, _ in saved))
                del st.session_state["batch_frames"]
                del st.session_state["batch_segments"]
                st.rerun()


def main():
    st.title("🤟 Sign Language → Text")
    st.caption("Upload a video, get an English transcript. Powered by MediaPipe "
               "landmark tracking + DTW template matching (see README for how it works).")

    sample_fps, motion_threshold, use_claude, api_key = render_sidebar()

    tab1, tab2, tab3 = st.tabs(["Translate a Video", "Build Vocabulary", "Batch Add"])
    with tab1:
        render_translate_tab(sample_fps, motion_threshold, use_claude, api_key)
    with tab2:
        render_vocabulary_tab(sample_fps)
    with tab3:
        render_batch_add_tab(sample_fps, motion_threshold)


if __name__ == "__main__":
    main()
