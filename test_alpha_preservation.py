import io
import unittest
from PIL import Image
from alpha_preservation import preserve_alpha


class AlphaPreservationTest(unittest.TestCase):
    def test_transparent_center_gradient_and_rgb_roundtrip(self):
        image = Image.new("RGBA", (4, 4))
        pixels = [(210, 40, 80, a) for a in [0, 64, 128, 255] * 4]
        image.putdata(pixels)
        model = Image.new("L", image.size, 192)
        alpha, output = preserve_alpha(image, model)
        buffer = io.BytesIO()
        output.save(buffer, format="PNG")
        decoded = Image.open(io.BytesIO(buffer.getvalue()))
        self.assertEqual(list(alpha.getdata()), [0, 64, 128, 192] * 4)
        self.assertEqual(list(decoded.convert("RGB").getdata()), [p[:3] for p in pixels])

    def test_min_avoids_double_attenuation(self):
        image = Image.new("RGBA", (16, 16), (255, 80, 40, 128))
        alpha, _ = preserve_alpha(image, Image.new("L", image.size, 128))
        self.assertEqual(alpha.getpixel((0, 0)), 128)
        self.assertEqual(round(128 * 128 / 255), 64)

    def test_opaque_rgb_and_rgba_are_identity(self):
        model = Image.new("L", (4, 4))
        model.putdata(range(0, 256, 16))
        for mode in ("RGB", "RGBA"):
            source = Image.new(mode, (4, 4), (230, 40, 10) if mode == "RGB" else (230, 40, 10, 255))
            alpha, output = preserve_alpha(source, model)
            self.assertEqual(alpha.tobytes(), model.tobytes())
            self.assertEqual(output.convert("RGB").tobytes(), source.convert("RGB").tobytes())

    def test_palette_transparency(self):
        source = Image.new("P", (4, 4), 0)
        source.info["transparency"] = 0
        alpha, _ = preserve_alpha(source, Image.new("L", source.size, 255))
        self.assertEqual(alpha.getextrema(), (0, 0))

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            preserve_alpha(Image.new("RGB", (4, 4)), Image.new("L", (2, 2)))


if __name__ == "__main__":
    unittest.main()
