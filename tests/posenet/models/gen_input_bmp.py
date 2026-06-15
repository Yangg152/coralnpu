# gen_input_bmp.py - 运行一次即可生成 bmp 文件
from PIL import Image

img = Image.open('test_couple.jpg').convert('RGB')
img = img.resize((481, 353), Image.BILINEAR)  # width=481, height=353
img.save('test_input_353x481.bmp', format='BMP')
print(f"Generated BMP file: {img.size[0]}x{img.size[1]}")
