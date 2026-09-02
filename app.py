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


def main():
    st.title("🤟 Sign Language → Text")
    st.caption("Upload a video, get an English transcript. Powered by MediaPipe "
               "landmark tracking + DTW template matching (see README for how it works).")

    sample_fps, motion_threshold, use_claude, api_key = render_sidebar()

    tab1, tab2 = st.tabs(["Translate a Video", "Build Vocabulary"])
    with tab1:
        render_translate_tab(sample_fps, motion_threshold, use_claude, api_key)
    with tab2:
        render_vocabulary_tab(sample_fps)


if __name__ == "__main__":
    main()
