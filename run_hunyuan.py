# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the respective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

import argparse
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.environ.get('HF_HOME', '/data2/hja/CKPT/Hunyuan3D-2')
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'

from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


def main():
    parser = argparse.ArgumentParser(description='Run Hunyuan3D-2 image-to-3D; infer output format from output_path extension')
    parser.add_argument('--input_path', type=str, required=True, help='input image path')
    parser.add_argument('--output_path', type=str, default='./tmp/hunyuan_mesh.glb', help='output path (.glb or .obj)')
    parser.add_argument('--model_path', type=str, default='tencent/Hunyuan3D-2', help='model path or HuggingFace model name')
    parser.add_argument('--seed', type=int, default=None, help='random seed (optional)')
    args = parser.parse_args()

    # 加载并预处理图像（与 minimal_demo.py 保持一致）
    image = Image.open(args.input_path).convert("RGBA")
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)

    # 加载 pipeline
    print(f"[Hunyuan3D] Loading shape generation pipeline from {args.model_path}...")
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_path)

    # 生成 3D 网格
    print(f"[Hunyuan3D] Generating 3D mesh from image...")
    kwargs = {'image': image}
    if args.seed is not None:
        kwargs['seed'] = args.seed
    
    mesh = pipeline_shapegen(**kwargs)[0]
    print(f"[Hunyuan3D] Mesh generation completed!")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)

    # 从输出文件扩展名推断格式并导出
    ext = os.path.splitext(args.output_path)[1].lower().lstrip('.')
    if ext == 'glb':
        mesh.export(args.output_path)
        print(f"[Hunyuan3D] Saved GLB to {args.output_path}")
    elif ext == 'obj':
        mesh.export(args.output_path)
        print(f"[Hunyuan3D] Saved OBJ to {args.output_path}")
    else:
        raise ValueError(f"Unsupported output extension '.{ext}'. Use .glb or .obj.")


if __name__ == '__main__':
    main()

