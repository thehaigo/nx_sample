# NxNn

**TODO: Add description**

Nx,exlaを使用して以下の書籍3章ニューラルネットワークを実装したサンプルコードです
https://www.oreilly.co.jp/books/9784873117584/

mnistのファイルをダウンロードして、
Pure NxとXLA backend Nxのベンチマークを走らせます

```
iex -s mix
iex(1)> Dataset.download(:mnist)
iex(2)> NxNn.acc
```
