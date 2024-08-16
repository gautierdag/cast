import torch
import torchvision.transforms as T
from transformers import (
    AutoModel,
    AutoTokenizer
)
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InternVLModel:
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "OpenGVLab/InternVL2-8B",
            torch_dtype=torch.float16,  # float32 for cpu
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "OpenGVLab/InternVL2-8B", trust_remote_code=True
        )
        self.model.eval()

    def generate(
        self, instructions: str, image=None, max_new_tokens=256, start_decode: str = ""
    ) -> str:
        """
        Given an instruction and an optional image, generate a response of maximum length `max_new_tokens`.

        Start decode is a string that will be appended after the instructions.
        For instance to start the assistant response.
        """
        if image is not None:
            ## in case of multi-image inference
            if isinstance(image, torch.Tensor):
                image = image.to(self.model.device)
            else:
                image = self.load_image(image)
            msgs = instructions.replace("<image>", "") + "\n" + start_decode
        else:
            msgs = instructions + "\n" + start_decode

        generation_config = dict(
            num_beams=1,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        output = self.model.chat(
            self.tokenizer,
            image,
            msgs,
            generation_config,
            history=None, 
            return_history=False,
        )
        return output
    
    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform


    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio


    def dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images


    def load_image(self, image_file, input_size=448, max_num=6):
        if isinstance(image_file, str):
            image = Image.open(image_file).convert('RGB')
        else:
            image = image_file.convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(torch.float16).to(self.model.device)
        return pixel_values
    

    def get_concat_h(self, image_file, image_file_2):
        image1 = self.load_image(image_file, max_num=6)
        image2 = self.load_image(image_file_2, max_num=6)
        return torch.cat((image1, image2), dim=0)