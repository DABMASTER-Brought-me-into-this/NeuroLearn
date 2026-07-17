import streamlit as st
import os
import tempfile
import shutil
import logic

st.title("NeuroLearn: AI Active Recall")

# Inputs
uploaded_file = st.file_uploader("Upload Lecture Slides", type=["pptx", "pdf"])
deck_name = st.text_input("Desired Deck Name", value="NeuroLearn_Deck")

# Execution
if uploaded_file and st.button("Generate Deck"):
    with st.spinner("Running Neural Pipelines... This may take a few minutes."):
        # Create a temporary directory (The "Virtual Folder")
        with tempfile.TemporaryDirectory() as temp_dir:

            # Save the uploaded file to this folder
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Trigger the formatting function (which now runs the pipeline internally)
                final_csv = logic.create_csv_file(deck_name, temp_path, temp_dir)

                # Zip Everything for Download
                if final_csv:
                    zip_name = f"{deck_name}_package"
                    shutil.make_archive(zip_name, 'zip', temp_dir)

                    with open(f"{zip_name}.zip", "rb") as f:
                        st.success("Deck Generated Successfully!")
                        st.download_button(
                            label="Download NeuroLearn Deck (ZIP)",
                            data=f,
                            file_name=f"{zip_name}.zip",
                            mime="application/zip"
                        )
                else:
                    st.error("No cards were generated. Check your slide content.")

            except Exception as e:
                st.error(f"Pipeline Error: {e}")