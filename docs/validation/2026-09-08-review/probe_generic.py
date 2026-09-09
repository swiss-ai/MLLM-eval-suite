"""Tiny real-engine parity check of pre/post Molmo routing fix, not a benchmark."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['WANDB_MODE'] = 'disabled'
    root = Path(__file__).parent / 'lmms'
    sys.path.insert(0, str(root))
    from PIL import Image
    import vllm
    from lmms_eval.api.instance import Instance
    from lmms_eval.models.simple import vllm as wrapper

    baseline_sha = 'e8e1b2accd0603996b717130ad20ebca89c94efd'
    baseline_source = subprocess.check_output(['git', '-C', str(root), 'show', f'{baseline_sha}:lmms_eval/models/simple/vllm.py'], text=True)
    tree = ast.parse(baseline_source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'VLLM')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'generate_until')
    namespace = dict(vars(wrapper))
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<baseline-generate-until>', 'exec'), namespace)
    baseline_generate = namespace['generate_until']
    started = time.time()
    model = wrapper.VLLM(model=args.model, batch_size=2, gpu_memory_utilization=0.45,
                         max_new_tokens=16, max_model_len=4096, enforce_eager=True,
                         enable_prefix_caching=False, max_num_seqs=2,
                         disable_log_stats=True, enable_thinking=False,
                         limit_mm_per_prompt={'image': 1, 'video': 0})
    images = [Image.new('RGB', (224, 224), color) for color in ('red', 'blue')]
    model.task_dict = {'generic_image_probe': {'test': [{'image': image} for image in images]}}
    requests = [Instance(request_type='generate_until', idx=idx,
                arguments=('Name the dominant color in this image. Answer with one word.',
                           {'max_new_tokens':16, 'temperature':0, 'top_p':1},
                           lambda doc: [doc['image']], idx, 'generic_image_probe', 'test'),
                metadata={'task':'generic_image_probe', 'doc_id':idx, 'repeats':1})
                for idx in range(len(images))]
    before = baseline_generate(model, requests)
    after = model.generate_until(requests)
    result = {'model':args.model, 'baseline':baseline_sha,
              'candidate':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
              'vllm':vllm.__version__, 'requests':len(requests), 'before':before, 'after':after,
              'exact_matches':sum(a == b for a,b in zip(before,after)),
              'nonempty':all(bool(text.strip()) for text in after),
              'image_sha256':[hashlib.sha256(image.tobytes()).hexdigest() for image in images],
              'elapsed_seconds':time.time()-started,
              'scope':'Actual simple VLLM wrapper, same loaded engine, synthetic images; no accuracy or speed claim.'}
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    assert result['exact_matches'] == len(requests) and result['nonempty']


if __name__ == '__main__':
    main()
