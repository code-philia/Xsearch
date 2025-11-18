
import os
import torch
import argparse
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (RobertaModel, RobertaTokenizer, RobertaTokenizerFast)
from dataloader import TextDataset, textdataset_collate_fn, remove_comments_and_docstrings
from model import Model
import multiprocessing
import numpy as np
import re
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
import html

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Alignment</title>
  <style>
    body { font-family: sans-serif; padding: 1rem; background: #fafafa; }
    h3 { margin-top: 2rem; color: #333; }
    pre.comment, pre.code {
      white-space: pre; /* preserve spaces & newlines */
      font-family: monospace;
      line-height: 1.4;
      margin: 1em 0;
      position: relative;
      padding: 1rem;
      border-radius: 6px;
    }
    pre.comment { background: #ffffff; border: 1px solid #b6e0fe; }
    pre.code    { background: #ffffff; border: 1px solid #f1d3a3; }
    #infoBox {
      position: absolute;
      background: white;
      border: 1px solid #666;
      padding: .5em;
      box-shadow: 0 0 10px rgba(0,0,0,.2);
      display: none;
      pointer-events: none;
      font-size: .9em;
      z-index: 10;
      border-radius: 4px;
    }
    #overlay {
      position: absolute;
      top: 0; left: 0;
      width: 100%; height: 100%;
      pointer-events: none;
      z-index: 5;
    }
    .ground-truth {
      margin: 2em 0;
      padding: 1em;
      border-radius: 6px;
      background: #ffffff;
      border: 1px solid #ddd;
    }
    .ground-truth h3 { margin-top: 0; }
    .concept-block { margin-top: 1em; }
    .concept-block strong { display: block; margin-bottom: .5em; }
    .concept-block .nl, .concept-block .code {
      margin-left: 1em;
      margin-bottom: .5em;
    }
    .concept-block span {
      display: inline-block;
      margin: 0 .3em;
      padding: .2em .4em;
      border-radius: 3px;
    }
    .concept-nl-token { background: #def; }
    .concept-code-token { background: #fed; }
  </style>
</head>
<body>
  <div id="infoBox"></div>
  <svg id="overlay"></svg>
  <h3>Predictions</h3>

  {{ highlighted_comment | safe }}
  {{ highlighted_code    | safe }}
  
  <h3>Ground Truth</h3>
  <div class="ground-truth">
    {% for concept in concept_list %}
      <div class="concept-block">
        <strong>Concept {{ loop.index }}:</strong>
        <div class="nl">
          <em>Comment:</em>
          {% for s in concept.nl %}
            <span class="concept-nl-token">{{ s }}</span>
          {% endfor %}
        </div>
        <div class="code">
          <em>Code:</em>
          {% for s in concept.code %}
            <span class="concept-code-token">{{ s }}</span>
          {% endfor %}
        </div>
      </div>
    {% endfor %}
  </div>

  <script>
      // 把 Python 传过来的映射也序列化到前端
      const top5Map = {{ top5_map | tojson }};
      const commentEls = document.querySelectorAll('span.comment-token');
      const codeEls    = document.querySelectorAll('span.code-token');
      const infoBox    = document.getElementById('infoBox');

      function clearHighlights() {
        infoBox.style.display = 'none';
      }

      commentEls.forEach(el => {
        el.addEventListener('mouseenter', () => {
          clearHighlights();
          const ci = +el.dataset.idx;
          const entry = top5Map[ci];
          if (!entry) return;

          // 在 InfoBox 中列出 Top-5
          const htmlLines = entry.ids.map((cj, rank) =>
            `<div>${rank+1}. <b>${entry.tokens[rank]}</b> — ${entry.sims[rank].toFixed(3)}</div>`
          ).join("");
          infoBox.innerHTML =
            `<b>Comment:</b> ${el.textContent}<br>` +
            `<b>Top 5 matches:</b><br>` + htmlLines;

          // 定位并显示
          const r = el.getBoundingClientRect();
          infoBox.style.top  = (r.bottom + window.scrollY + 5) + 'px';
          infoBox.style.left = (r.left   + window.scrollX)  + 'px';
          infoBox.style.display = 'block';
        });

        el.addEventListener('mouseleave', () => {
          clearHighlights();
        });
      });
</script>
</body>
</html>
"""

def get_latest_epoch(folder: str) -> str:
    # list all entries named “Epoch_<num>”
    epochs = [
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d)) and d.startswith("Epoch_")
    ]
    # pick the one with the highest trailing number
    latest = max(epochs, key=lambda x: int(x.split("_", 1)[1]))
    return latest


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")

    parser.add_argument("--lang", default=None, type=str,
                        help="language.")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")

    args = parser.parse_args()
    pool = multiprocessing.Pool(16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaModel.from_pretrained(args.model_name_or_path)
    model = Model(model)

    output_dir = os.path.join(args.output_dir, get_latest_epoch(args.output_dir), "model_path.pth")
    model.load_state_dict(torch.load(output_dir), strict=False)
    model.to(args.device)
    model.eval()

    # Load labeled samples first
    full_dataset = TextDataset(tokenizer, args, args.test_data_file, pool)
    train_sampler = SequentialSampler(full_dataset)
    train_dataloader = DataLoader(full_dataset,
                                  sampler=train_sampler,
                                  batch_size=1,
                                  num_workers=4,
                                  collate_fn=textdataset_collate_fn)


    for step, batch in enumerate(train_dataloader):
        if step <= 1: continue
        # Get inputs
        code_inputs_ids = batch[0].to(args.device)
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        nl_inputs_ids = batch[3].to(args.device)
        ori2cur_pos = batch[4].to(args.device)
        match_list = batch[5]

        raw_comment = full_dataset.raw_comment[step]
        raw_code    = full_dataset.raw_code[step]
        raw_code = remove_comments_and_docstrings(raw_code, lang=args.lang)

        with torch.inference_mode():
            outputs = model(code_inputs=code_inputs_ids, attn_mask=attn_mask, position_idx=position_idx, nl_inputs=nl_inputs_ids)  # (batch_size, seq_length_code, hidden_size)
            code_probs,  nl_probs  = outputs.code_scores, outputs.nl_scores
            sim_mat = outputs.sim_mat

        batch_size = code_probs.size(0)
        for b in range(batch_size):
            # 1) Predicted sets
            pred_code = set((code_probs[b] > 0.5).nonzero(as_tuple=True)[0].tolist())
            pred_nl   = set((nl_probs[b] > 0.5).nonzero(as_tuple=True)[0].tolist())

            code_tokens = tokenizer.convert_ids_to_tokens(batch[0][b])
            nl_tokens = tokenizer.convert_ids_to_tokens(batch[3][b])
            sim = sim_mat[b].cpu().numpy()
            nl_tokens = [t.lstrip("Ġ") for t in nl_tokens]
            code_tokens = [t.lstrip("Ġ") for t in code_tokens]

            # 2) Ground-truth sets (flatten all spans)
            concept_list = []
            for comment_span, code_span in match_list[b]:
                this_nl_strs = []
                this_code_strs = []
                # 把一个 concept 里所有连续的 nl token 段拼成字符串
                for j in range(0, len(comment_span), 2):
                    idx0, idx1 = comment_span[j], comment_span[j + 1]
                    tok_slice = nl_tokens[idx0: idx1 + 1]
                    this_nl_strs.append(tokenizer.convert_tokens_to_string(tok_slice))
                # 把一个 concept 里所有连续的 code token 段拼成字符串
                for j in range(0, len(code_span), 2):
                    idx0, idx1 = code_span[j], code_span[j + 1]
                    tok_slice = code_tokens[idx0: idx1 + 1]
                    this_code_strs.append(tokenizer.convert_tokens_to_string(tok_slice))
                concept_list.append({
                    "nl": this_nl_strs,
                    "code": this_code_strs
                })

            env = Environment(
                trim_blocks=True,
                lstrip_blocks=True,
                )
            env.globals['html'] = html
            env.globals['re'] = re
            env.globals['enumerate'] = enumerate

            # highlight comment
            escaped_comment = html.escape(raw_comment)
            highlighted_comment = '<pre class="comment">' + escaped_comment + '</pre>'
            top5_map = {}
            for idx in pred_nl:
                tk = nl_tokens[idx]
                if not re.fullmatch(r"[A-Za-z0-9]+", tk):
                    continue
                pat = re.compile(rf'(?<!\w){re.escape(tk)}(?!\w)', flags=re.IGNORECASE)
                highlighted_comment = pat.sub(
                    f'<span class="comment-token" data-idx="{idx}" style="background:#def;">\g<0></span>',
                    highlighted_comment
                )
                top5_ids = np.argsort(sim[idx])[::-1][:5].tolist()
                top5_sims = [float(sim[idx][j]) for j in top5_ids]
                top5_map[idx] = {
                    "ids": top5_ids,
                    "sims": top5_sims,
                    "tokens": [code_tokens[j] for j in top5_ids]  # 可选：把 token 文本也传过去
                }

            # highlight code
            escaped_code = html.escape(raw_code)
            highlighted_code = '<pre class="code">' + escaped_code + '</pre>'
            for idx in pred_code:
                tk = code_tokens[idx]
                if tk == "" or tk == "_":
                    continue
                esc = re.escape(tk)
                if re.fullmatch(r"[A-Za-z0-9_]+", tk):
                    pat_code = re.compile(rf'(?<![A-Za-z0-9_]){esc}(?![A-Za-z0-9_])')
                else:
                    pat_code = re.compile(esc)
                highlighted_code = pat_code.sub(
                    f'<span class="code-token" data-idx="{idx}" style="background:#fed;">\g<0></span>',
                    highlighted_code
                )

            template = env.from_string(TEMPLATE)
            html_output = template.render(
                highlighted_comment=highlighted_comment,
                highlighted_code=highlighted_code,
                top5_map=top5_map,
                concept_list=concept_list,
            )

            with open("debug.html", 'w', encoding='utf-8') as f:
                f.write(html_output)

            exit()

if __name__ == "__main__":
    main()

 