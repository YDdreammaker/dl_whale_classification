# ๐ฅHappyWhale - Dolphin and Whale

<div style="text-align:center"><img src=./img/main.png?raw=true /></div>
<br>

# Contents

#### &nbsp;&nbsp;&nbsp;&nbsp;**[๐งTask Description](https://github.com/YDdreammaker/dl_whale_classification#task-description-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[๐Project Result](https://github.com/YDdreammaker/dl_whale_classification#project-result-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[โInstallation](https://github.com/YDdreammaker/dl_whale_classification#installation-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[๐นCommand Line Interface](https://github.com/YDdreammaker/dl_whale_classification#command-line-interface-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[๐คCollaboration Tools](https://github.com/YDdreammaker/dl_whale_classification#collaboration-tools-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[๐ฉโ๐ฆโ๐ฆWho Are We?](https://github.com/YDdreammaker/dl_whale_classification#who-are-we-1)**
<br>

# Task Description

### Subject https://www.kaggle.com/competitions/happy-whale-and-dolphin 
<br>
์ด๋ฒ ๋ํ์ ์ฃผ์ ๋ ๋๊ณ ๋์ ๊ณ ๋์ ์ฌ์ง์ผ๋ก individual_id ๋ถ๋ฅํ๋ ๋ฌธ์ ์์ต๋๋ค.  ๋๊ณ ๋์ ๊ณ ๋์ ์ง๋๋ฌ๋ฏธ์ ์ฌ๋์ ์ง๋ฌธ๊ณผ ๊ฐ์ด ๊ฐ ๊ฐ์ฒด๋ฅผ ๋ถ๋ฅํ  ์ ์๋ ํน์ง์ด ์๋ค๊ณ  ์๊ฐํด ํด๋น ๋ฌธ์ ๋ฅผ object recognition ์ด๋ก ์ผ๋ก ์ ๊ทผํ์์ต๋๋ค.

<br>
๋๊ณ ๋์ ์ฌ์ง์ ์ง๋๋ฌ๋ฏธ์ ์ง๋๋ฌ๋ฏธ๋ฅผ ํฌํจํ ๋ชธํต์ผ๋ก object detection ์งํํ๊ณ , ํด๋น ์ด๋ฏธ์ง๋ฅผ ๊ฐ์ง๊ณ  ๊ฐ ๊ฐ์ฒด๋ฅผ ๋ถ๋ฅํ๋ ๋ชจ๋ธ์ ์์ฑํ์์ต๋๋ค.
<br>

### Data

- ํ๋ จ ๋ฐ์ดํฐ : 51033์ฅ์ ์ด๋ฏธ์ง์ ํด๋น ์ด๋ฏธ์ง์ ์ข๊ณผ individual_id

- ํ์คํธ ๋ฐ์ดํฐ : 27916์ฅ์ ์ด๋ฏธ์ง
<br>

### Metric

<img src="./img/metric.png" />
</br>

<img src="./img/metric_score.png" width='300px' height='180px' />
</br>

# Project Result

<div><img src=./img/rank.png?raw=true /></div>

- ์๋ฉ๋ฌ 47 ๋ฑ / 1,613 ํ

- Public LB Score: 0.85147 / Private LB Score: 0.81686

- ์๋ฃจ์์ [์ด๊ณณ](https://www.notion.so/Solution-3ccc8fd2a39841a78d5726946b109707)์์ ํ์ธํ์ค ์ ์์ต๋๋ค.
</br>

# Installation

```bash
# clone repository
git clone https://github.com/YDdreammaker/dl_whale_classification.git

# install necessary tools
pip install -r requirements.txt
```

### Dataset Structure

```shell
[dataset]/
โโโ train.csv
โโโ sample_submission.csv
โโโ train/
    โโโ *.jpg
    โโโ ...
    โโโ *.jpg
โโโ test/
    โโโ *.jpg
    โโโ ...
    โโโ *.jpg
```

### Code Structure

```shell
[code]
โโโ data.py/ # modules for dataset
โโโ loss.py/ # modules for loss during train
โโโ schedulers.py/ # Cosine Annealing Warmup scheduler
โโโ utils.py/ # useful utilities
โโโ model.py/ # modules for model during train
โโโ train.py/ # modules for train and validation of one epoch
โโโ README.md
โโโ requirements.txt
โโโ main.py
โโโ inference.py
```



# Command Line Interface

## Train

```bash
$ python main.py --n_split \
                 --fold \
                 --seed \
                 --epoch \
                 --model_name \
                 --train_data \
                 --train_image_root \
                 --image_size \
                 --batch_size \
                 --iters_to_accumulate \
                 --learning_rate \         # max learning rate for scheduler
                 --emb_size \              # embedding size
                 --margin \
                 --s \                    # ArcFace parameter
                 --m \                    # ArcFace parameter
                 --weight_decay \
                 --experiment \           # for keep training for ex-train
```

## Inference

#### make embedding pt file
```shell
$ python inference.py --inference_type embedding
```

#### make logit pt file
```shell
$ python inference.py --inference_type logit
```
</br>

# Collaboration Tools
<table>
    <tr height="200px">
        <td align="center" width="350px">	
            <a href="https://www.notion.so/b47246b96c204ca38f96c45888919525?v=f2ab615cde7342c78d3761641a828e5c"><img height="180px" width="320px" src="./img/notion.png?raw=true"/></a>
            <br />
            <a href="https://www.notion.so/b47246b96c204ca38f96c45888919525?v=f2ab615cde7342c78d3761641a828e5c">Notion</a>
        </td>
        <td align="center" width="350px">	
            <a><img height="180px" width="320px" src="./img/wandb.png?raw=true"/></a>
            <br />
            <a>WanDB</a>
        </td>
    </tr>
</table>
</br>

# Who Are We?

<table>
    <tr height="140px">
        <td align="center" width="130px">	
            <a href="https://github.com/DaeYeongMonster"><img height="100px" width="100px" src="https://cdn.discordapp.com/attachments/949221955372974090/966225553348780092/KakaoTalk_20220419_160217458_02.jpg"/></a>
            <br />
            <a href="https://github.com/DaeYeongMonster">๊น๋์<br />eodudahs@gmail.com</a>
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/ahaampo5"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/60084351?v=4"/></a>
            <br />
            <a href="https://github.com/ahaampo5">๊น์ค์ฒ <br />ahaampo5@gmail.com</a>
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/thsckdduq"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/52391044?v=4"/></a>
            <br />
            <a href="https://github.com/thsckdduq">์์ฐฝ์ฝ<br />thsckdduq@gmail.com</a>
        </td>
    </tr>
    <tr height="140px">
        <td align="center" width="130px">
            <a href="https://github.com/NOTITLEUNTITLE"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/88427224?v=4"/></a>
            <br />
            <a href="https://github.com/NOTITLEUNTITLE">์ฉ๋ค์ด<br />inopiction@naver.com</a>
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/aperyear"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/88475399?v=4"/></a>
            <br />
            <a href="https://github.com/aperyear">์ด๋จ์ฃผ<br />aperyear@gmail.com</a>
        </td>
    </tr>
</table>