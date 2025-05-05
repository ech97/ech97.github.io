from PIL import Image

# 1) 원본 불러오기
orig = Image.open("assets/images/header.jpg")
w, h = orig.size

# 2) 새 캔버스 생성 (가로 5배)
canvas = Image.new("RGBA", (w*5, h), (15,61,59,255))  # 배경색: 흰색

# 3) 원본 이미지를 원하는 위치에 붙여넣기
# canvas.paste(orig, (0, 0))              # 왼쪽 정렬
canvas.paste(orig, (w*2, 0))         # 중앙 정렬 (원할 경우)

# 4) 저장
canvas.save("./output.png")