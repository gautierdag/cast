import openai
import base64
from PIL import Image
import json
import io


class GPT4OMini:
    def __init__(self):
        self.client = openai.Client()
        self.cache_location = "gpt4o-mini.cache.json"
        self.cache = {}
        try:
            with open(self.cache_location, "r") as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            pass

    @staticmethod
    def image_to_base64(image: Image) -> str:
        # Save the image to a bytes buffer
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        # Encode the bytes to a base64 string
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def generate(
        self, instructions: str, image=None, max_new_tokens=256, start_decode: str = ""
    ) -> str:
        """
        Given an instruction and an optional image, generate a response of maximum length `max_new_tokens`.

        Start decode is a string that will be appended after the instructions.
        For instance to start the assistant response.
        """
        if instructions in self.cache:
            return self.cache[instructions]

        # Remove the <image> tag from the instructions
        instructions = instructions.replace("<image>\n", "")

        message = {"role": "user", "content": [{"type": "text", "text": instructions}]}
        if image:
            base64_image = self.image_to_base64(image)
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
            message["content"].append(image_content)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[message],
            temperature=0,
            max_tokens=max_new_tokens,
        )
        response = response["choices"][0]["message"]["content"]

        self.cache[instructions] = response
        with open(self.cache_location, "w") as f:
            json.dump(self.cache, f)

        return response.choices[0].message.content
