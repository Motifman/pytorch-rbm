# pytorch-rbm
## 使い方
以下のコマンドを実行することでデフォルトのハイパーパラメータを用いてRBMの訓練が行われます
```
python3 train.py
```

引数を指定することで隠れ層のサイズや学習率の変更やGPUの使用などが可能です
```
python3 train.py --hidden_size 512 --lr 0.1 --device_id cuda:0
```

指定可能な引数を調べたい場合は以下のコマンドを実行するか、train.pyを確認してください
```
python3 train.py -h
```


## 結果
訓練途中の生成画像や再構成誤差をtensorboardXで確認することが可能です。以下のコマンドを実行してください
```
tensorboard --logdir log/
```

figディレクトリに以下のような結果が格納されます。上段は入力画像、中段は再構成画像、下段は差分画像になっています。
![reconst](fig/reconst_img.png)

また、平均自由エネルギーのプロットも保存されます。trainとtestの平均自由エネルギーの差をモニターすることで過学習の監視が可能です。
平均自由エネルギーは以下の式の計算結果をミニバッチで平均した値です
```math
\begin{align}
b(x) &= (b^{\mathrm{H}})^{\mathrm{T}} + x^{\mathrm{T}}W\\
F_{\theta}(x) &= -(b^{\mathrm{V}})^{\mathrm{T}}x - \sum^M_{i=1} \log(1 + \exp(b_i(x)))
\end{align}
```

## TODO
以下のオプションを実装予定です
+ [ ] 隠れ層の活性率の可視化 
+ [ ] 自由エネルギー、期待自由エネルギーによる回帰モデル
+ [ ] 深層ボルツマンマシン、動的ボルツマンマシン
+ [ ] 画像分類モデルの実装
