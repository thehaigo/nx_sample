# NxNn

**TODO: Add description**

Nx,exlaを使用して以下の書籍3章ニューラルネットワークを実装したサンプルコードです
https://www.oreilly.co.jp/books/9784873117584/

mnistのファイルをダウンロードして、
Pure NxとXLA backend Nxのベンチマークを走らせます

```
$ iex -S mix
iex(1)> Dataset.download(:mnist)
iex(2)> NxNn.acc

Name                     ips        average  deviation         median         99th %
exla cpu batch          1.34     0.0125 min     ±4.21%     0.0123 min     0.0132 min
exla cpu                0.25     0.0668 min     ±4.29%     0.0668 min     0.0688 min
nx                   0.00294       5.67 min     ±0.00%       5.67 min       5.67 min
nx batch             0.00271       6.14 min     ±0.00%       6.14 min       6.14 min

Comparison:
exla cpu batch          1.34
exla cpu                0.25 - 5.35x slower +0.0543 min
nx                   0.00294 - 454.65x slower +5.66 min
nx batch             0.00271 - 492.39x slower +6.13 min

```
