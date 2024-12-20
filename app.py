import argparse
import os
import tempfile
import zipfile

import gradio as gr
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image

TITLE = "WaifuDiffusion Tagger (Batch Processing)"
DESCRIPTION = """
This app allows you to tag anime-style images using various WaifuDiffusion tagger models. 
Upload multiple images or specify a directory containing images and get general tags, ratings, and character predictions for each.

Example image by [ほし☆☆☆](https://www.pixiv.net/en/users/43565085)
"""

# Model repositories (unchanged)
SWINV2_MODEL_DSV3_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
CONV_MODEL_DSV3_REPO = "SmilingWolf/wd-convnext-tagger-v3"
VIT_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-tagger-v3"
VIT_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
EVA02_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
MOAT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
SWIN_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"

MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# Kaomojis list (unchanged)
kaomojis = [
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-slider-step", type=float, default=0.05)
    parser.add_argument("--score-general-threshold", type=float, default=0.35)
    parser.add_argument("--score-character-threshold", type=float, default=0.85)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()

def load_labels(dataframe) -> list[str]:
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes

def mcut_threshold(probs):
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh

class Predictor:
    def __init__(self):
        self.model_target_size = None
        self.last_loaded_repo = None

    def download_model(self, model_repo):
        csv_path = huggingface_hub.hf_hub_download(
            model_repo,
            LABEL_FILENAME,
        )
        model_path = huggingface_hub.hf_hub_download(
            model_repo,
            MODEL_FILENAME,
        )
        return csv_path, model_path

    def load_model(self, model_repo):
        if model_repo == self.last_loaded_repo:
            return

        csv_path, model_path = self.download_model(model_repo)

        tags_df = pd.read_csv(csv_path)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        model = rt.InferenceSession(model_path)
        _, height, width, _ = model.get_inputs()[0].shape
        self.model_target_size = height

        self.last_loaded_repo = model_repo
        self.model = model

    def prepare_image(self, image):
        target_size = self.model_target_size

        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def predict(
        self,
        images,
        model_repo,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
    ):
        self.load_model(model_repo)

        results = []
        for image in images:
            prepared_image = self.prepare_image(image)

            input_name = self.model.get_inputs()[0].name
            label_name = self.model.get_outputs()[0].name
            preds = self.model.run([label_name], {input_name: prepared_image})[0]

            labels = list(zip(self.tag_names, preds[0].astype(float)))

            # First 4 labels are actually ratings: pick one with max
            ratings_names = [labels[i] for i in self.rating_indexes]
            rating = dict(ratings_names)

            # General tags
            general_names = [labels[i] for i in self.general_indexes]

            if general_mcut_enabled:
                general_probs = np.array([x[1] for x in general_names])
                general_thresh = mcut_threshold(general_probs)

            general_res = [x for x in general_names if x[1] > general_thresh]
            general_res = dict(general_res)

            # Characters
            character_names = [labels[i] for i in self.character_indexes]

            if character_mcut_enabled:
                character_probs = np.array([x[1] for x in character_names])
                character_thresh = mcut_threshold(character_probs)
                character_thresh = max(0.15, character_thresh)

            character_res = [x for x in character_names if x[1] > character_thresh]
            character_res = dict(character_res)

            sorted_general_strings = sorted(
                general_res.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            sorted_general_strings = [x[0] for x in sorted_general_strings]
            sorted_general_strings = (
                ", ".join(sorted_general_strings).replace("(", "\(").replace(")", "\)")
            )

            results.append((sorted_general_strings, rating, character_res, general_res))

        return results

def main():
    args = parse_args()
    predictor = Predictor()

    dropdown_list = [
        SWINV2_MODEL_DSV3_REPO,
        CONV_MODEL_DSV3_REPO,
        VIT_MODEL_DSV3_REPO,
        VIT_LARGE_MODEL_DSV3_REPO,
        EVA02_LARGE_MODEL_DSV3_REPO,
        MOAT_MODEL_DSV2_REPO,
        SWIN_MODEL_DSV2_REPO,
        CONV_MODEL_DSV2_REPO,
        CONV2_MODEL_DSV2_REPO,
        VIT_MODEL_DSV2_REPO,
    ]

    with gr.Blocks(
        title=TITLE,
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"),
    ) as demo:
        gr.Markdown(
            value=f"<h1 style='text-align: center; margin-bottom: 1rem'>{TITLE}</h1>"
        )
        gr.Markdown(value=DESCRIPTION)

        # Inject custom CSS using gr.HTML
        gr.HTML(
            """
            <style>
            /* Custom CSS for enhanced styling */
            .gradio-container {
                background-color: #1a1a2e; /* Dark neon purple background */
                color: #eaeaea; /* Light gray text */
                font-family: 'Arial', sans-serif; /* Modern font */
            }

            h1, h2, h3, h4, h5, h6 {
                color: #eaeaea; /* Light gray text */
            }

            .gr-button {
                background-color: #4CAF50; /* Green button */
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 8px;
                cursor: pointer;
                transition: background-color 0.3s; /* Smooth hover transition */
            }

            .gr-button:hover {
                background-color: #3e8e41; /* Darker green on hover */
            }

            .gr-dropdown {
                background-color: #333; /* Dark gray dropdown */
                color: #eaeaea; /* Light gray text */
                border: 1px solid #555; /* Darker gray border */
                padding: 8px 12px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); /* Subtle shadow */
            }

            .gr-dropdown:hover {
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); /* Stronger shadow on hover */
            }

            .gr-slider {
                background-color: #333; /* Dark gray slider track */
                border-radius: 10px; /* Rounded slider track */
            }

            .gr-slider-thumb {
                background-color: #e94560; /* Neon pink thumb */
                border: 2px solid #555; /* Dark gray border */
                border-radius: 50%; /* Rounded thumb */
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3); /* Thumb shadow */
                cursor: pointer; /* Cursor change on hover */
            }

            .gr-checkbox {
                background-color: #333; /* Dark gray checkbox background */
                color: #eaeaea; /* Light gray text */
                border: 1px solid #555; /* Darker gray border */
                padding: 8px 12px;
                border-radius: 5px;
            }

            .gr-textbox {
                background-color: #333; /* Dark gray textbox background */
                color: #eaeaea; /* Light gray text */
                border: 1px solid #555; /* Darker gray border */
                padding: 8px 12px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); /* Subtle shadow */
            }

            .gr-textbox:hover {
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); /* Stronger shadow on hover */
            }

            .gr-dataframe {
                background-color: #2a2a72; /* Darker purple background */
                color: #eaeaea; /* Light gray text */
                border: 1px solid #555; /* Darker gray border */
                border-radius: 8px; /* Rounded corners */
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); /* Subtle shadow */
                max-height: 400px; /* Added height control */
                overflow-y: auto; /* Added scrolling */
            }

            .gr-gallery {
                border: 2px solid #d1d5db;
                border-radius: 8px;
                padding: 10px;
                background-color: #2a2a72; /* Darker purple */
                color: #eaeaea;
                overflow: auto;
            }

            .gr-gallery img {
                max-width: 100%;
                height: auto;
                border: 1px solid #555; /* Dark gray border */
                border-radius: 5px; /* Rounded corners for images */
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); /* Subtle shadow */
                cursor: pointer; /* Cursor change on hover */
                transition: transform 0.2s ease; /* Smooth scaling transition */
            }

            .gr-gallery img:hover {
                transform: scale(1.1); /* Image zoom on hover */
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); /* Stronger shadow on hover */
            }
            </style>
            """
        )

        with gr.Tabs(elem_id="image-tabs"):
            with gr.TabItem("Upload & Settings", elem_id="upload-tab"):
                with gr.Row():
                    with gr.Column(scale=1, elem_id="upload-column"):
                        # Option 1: Drag and drop images
                        image_upload = gr.Files(
                            label="Option 1: Drag and drop images",
                            file_types=["image"],
                            type="filepath",
                        )
                        images = gr.Gallery(
                            label="Uploaded Images",
                            show_label=True,
                            elem_id="gallery",
                            columns=3,
                            object_fit="contain",
                            height="auto",
                        )

                        # Option 2: Specify directory
                        input_directory = gr.Textbox(
                            label="Option 2: Input Directory for Images",
                            placeholder="Enter input directory path...",
                            lines=1,
                            interactive=True,
                            info="Enter the server-side directory path containing images. Ensure the application has read access to this directory.",
                        )

                        # Checkbox for saving .txt files
                        save_txt_files = gr.Checkbox(
                            label="Save results as .txt files",
                            value=False,
                            interactive=True,
                        )

                        # Output Directory Box (only visible when save_txt_files is checked)
                        output_directory = gr.Textbox(
                            label="Output Directory for .txt Files",
                            placeholder="Enter output directory path...",
                            lines=1,
                            interactive=True,
                            info="Enter the server-side directory path where .txt files will be saved. Ensure the application has write access to this directory.",
                            visible=False,
                        )

                        with gr.Accordion("Model Settings", open=False, elem_id="model-settings"):
                            model_repo = gr.Dropdown(
                                dropdown_list,
                                value=SWINV2_MODEL_DSV3_REPO,
                                label="Select Model",
                                interactive=True,
                            )
                            with gr.Row():
                                with gr.Column(scale=1):
                                    general_thresh = gr.Slider(
                                        0,
                                        1,
                                        step=args.score_slider_step,
                                        value=args.score_general_threshold,
                                        label="General Tags Threshold",
                                        info="Threshold for general tag confidence.",
                                    )
                                    general_mcut_enabled = gr.Checkbox(
                                        value=False,
                                        label="Use Adaptive Threshold for General Tags",
                                    )
                                with gr.Column(scale=1):
                                    character_thresh = gr.Slider(
                                        0,
                                        1,
                                        step=args.score_slider_step,
                                        value=args.score_character_threshold,
                                        label="Character Tags Threshold",
                                        info="Threshold for character tag confidence.",
                                    )
                                    character_mcut_enabled = gr.Checkbox(
                                        value=False,
                                        label="Use Adaptive Threshold for Character Tags",
                                    )

                        submit = gr.Button(
                            value="Process Images",
                            variant="primary",
                            size="large",
                            elem_id="submit-button",
                        )

            with gr.TabItem("Results", elem_id="results-tab"):
                with gr.Row():
                    with gr.Column(scale=1, elem_id="results-column"):
                        gr.Markdown(
                            value="### General Tags",
                            elem_id="general-tags-header",
                        )
                        output_general_tags = gr.Textbox(
                            placeholder="General tags will appear here...",
                            lines=20,
                            interactive=False,
                            show_label=False,
                            elem_id="general-tags-box",
                        )
                        # Output Directory Status
                        output_directory_status = gr.Textbox(
                            label="Output Directory Status",
                            placeholder="Status of the output directory...",
                            lines=2,
                            interactive=False,
                            show_label=False,
                        )
                    with gr.Column(scale=2, elem_id="results-column-2"):
                        gr.Markdown(
                            value="### Categorized Results",
                            elem_id="categorized-results-header",
                        )
                        output_dataframe = gr.Dataframe(
                            headers=["Image"],
                            label="Detailed Results",
                            wrap=True,
                            interactive=False
                        )

        def update_gallery(files):
            if files:
                return [file for file in files]
            return []

        def toggle_output_directory(save_txt):
            return gr.update(visible=save_txt)

        def process_images(image_files, input_dir, save_txt, output_dir, model_repo, general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled):
            images = []
            filenames = []

            # Handle files uploaded via drag-and-drop
            if image_files:
                for file in image_files:
                    try:
                        img = Image.open(file).convert("RGBA")
                        images.append(img)
                        filenames.append(os.path.basename(file))
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

            # Handle directory upload (server-side)
            if input_dir:
                if os.path.isdir(input_dir):
                    for filename in os.listdir(input_dir):
                        file_path = os.path.join(input_dir, filename)
                        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                            try:
                                img = Image.open(file_path).convert("RGBA")
                                images.append(img)
                                filenames.append(filename)
                            except Exception as e:
                                print(f"Error processing {file_path}: {e}")
                else:
                    print(f"Directory not found: {input_dir}")

            if not images:
                return [[]], "No images to process.", "No images found to process."

            results = predictor.predict(
                images,
                model_repo,
                general_thresh,
                general_mcut_enabled,
                character_thresh,
                character_mcut_enabled,
            )

            dataframe_data = []
            general_tags_list = []
            directory_status = ""

            if save_txt:
                if not output_dir:
                    directory_status = "No output directory specified. .txt files will not be saved."
                else:
                    if not os.path.exists(output_dir):
                        try:
                            os.makedirs(output_dir)
                            directory_status = f"Created directory: {output_dir}"
                        except Exception as e:
                            directory_status = f"Failed to create directory {output_dir}: {e}"
                            return [[]], "No images processed.", directory_status
                    else:
                        directory_status = f"Using existing directory: {output_dir}"

            for filename, result in zip(filenames, results):
                general_tags, rating, characters, _ = result
                top_rating = max(rating.items(), key=lambda x: x[1])[0]
                top_characters = ", ".join(sorted(characters.keys(), key=lambda x: characters[x], reverse=True)[:5])
                dataframe_data.append([filename])
                combined_tags = f"{general_tags}, {top_characters}, {top_rating}"
                general_tags_list.append(combined_tags)

                if save_txt and output_dir:
                    txt_filename = os.path.splitext(filename)[0] + ".txt"
                    txt_path = os.path.join(output_dir, txt_filename)
                    try:
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(combined_tags)
                    except Exception as e:
                        print(f"Error writing {txt_path}: {e}")

            general_tags_str = "\n\n".join(general_tags_list)
            return dataframe_data, general_tags_str, directory_status

        image_upload.change(update_gallery, inputs=[image_upload], outputs=[images])
        save_txt_files.change(toggle_output_directory, inputs=[save_txt_files], outputs=[output_directory])

        submit.click(
            process_images,
            inputs=[
                image_upload,
                input_directory,
                save_txt_files,
                output_directory,
                model_repo,
                general_thresh,
                general_mcut_enabled,
                character_thresh,
                character_mcut_enabled,
            ],
            outputs=[
                output_dataframe,
                output_general_tags,
                output_directory_status
            ],
        )

    demo.queue(max_size=10)
    demo.launch(share=True)

if __name__ == "__main__":
    main()
