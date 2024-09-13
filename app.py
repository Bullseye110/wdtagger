import argparse
import os

import gradio as gr
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image

TITLE = "WaifuDiffusion Tagger (Batch Processing)"
DESCRIPTION = """
This app allows you to tag anime-style images using various WaifuDiffusion tagger models. 
Upload multiple images and get general tags, ratings, and character predictions for each.

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

            # First 4 labels are actually ratings: pick one with argmax
            ratings_names = [labels[i] for i in self.rating_indexes]
            rating = dict(ratings_names)

            # Then we have general tags: pick any where prediction confidence > threshold
            general_names = [labels[i] for i in self.general_indexes]

            if general_mcut_enabled:
                general_probs = np.array([x[1] for x in general_names])
                general_thresh = mcut_threshold(general_probs)

            general_res = [x for x in general_names if x[1] > general_thresh]
            general_res = dict(general_res)

            # Everything else is characters: pick any where prediction confidence > threshold
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
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
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
            #gallery {
                border: 2px solid #d1d5db;
                border-radius: 8px;
                padding: 10px;
            }
            #submit-button {
                background-color: #6366f1;
                color: white;
                border-radius: 8px;
                margin-top: 1em;
                padding: 10px 20px;
            }
            #categorized-results-header, #general-tags-header {
                color: #4f46e5;
                margin-bottom: 0.5em;
            }
            #general-tags-box {
                border: 1px solid #d1d5db;
                border-radius: 5px;
                padding: 10px;
                background-color: #f9fafb;
                white-space: pre-wrap; /* Preserve formatting */
                overflow: auto;
            }
            .gradio-container {
                max-width: 1200px;
                margin: auto;
                padding: 20px;
            }
            </style>
            """
        )

        with gr.Tabs():
            with gr.TabItem("Upload & Settings"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_upload = gr.File(
                            file_count="multiple",
                            label="Upload Images",
                            file_types=["image"],
                        )
                        images = gr.Gallery(
                            label="Uploaded Images",
                            show_label=True,
                            elem_id="gallery",
                            columns=3,
                            object_fit="contain",
                            height="auto",
                        )

                        with gr.Accordion("Model Settings", open=False):
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

            with gr.TabItem("Results"):
                with gr.Row():
                    with gr.Column(scale=1):
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
                    with gr.Column(scale=2):
                        gr.Markdown(
                            value="### Categorized Results",
                            elem_id="categorized-results-header",
                        )
                        output_dataframe = gr.Dataframe(
                            headers=["Image"],
                            label="Detailed Results",
                            wrap=True,
                            interactive=False,
                            height=400,
                        )

        def update_gallery(files):
            if files:
                return [file.name for file in files]
            return []

        def process_images(image_files, model_repo, general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled):
            images = [Image.open(file.name).convert("RGBA") for file in image_files]
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
            for file, result in zip(image_files, results):
                general_tags, rating, characters, _ = result
                top_rating = max(rating.items(), key=lambda x: x[1])[0]
                top_characters = ", ".join(sorted(characters.keys(), key=lambda x: characters[x], reverse=True)[:5])
                dataframe_data.append([
                    os.path.basename(file.name),
                ])
                # Combine General Tags, Characters, and Rating
                combined_tags = f"{general_tags}, {top_characters}, {top_rating}"
                general_tags_list.append(combined_tags)  # Removed image name prefix
            
            # Add a blank line between tags from different images
            general_tags_str = "\n\n".join(general_tags_list)
            return dataframe_data, general_tags_str

        image_upload.change(update_gallery, inputs=[image_upload], outputs=[images])
        
        submit.click(
            process_images,
            inputs=[
                image_upload,
                model_repo,
                general_thresh,
                general_mcut_enabled,
                character_thresh,
                character_mcut_enabled,
            ],
            outputs=[
                output_dataframe,
                output_general_tags
            ],
        )

    demo.queue(max_size=10)
    demo.launch(share=True)

if __name__ == "__main__":
    main()
