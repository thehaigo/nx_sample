defmodule MnistNx do
  import Nx.Defn
  @defn_compiler {EXLA, max_float_type: {:f, 64}}
  defn predict(x, {w1,w2,w3,b1,b2,b3}) do
    x
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> sigmoid()
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> sigmoid()
    |> Nx.dot(w3)
    |> Nx.add(b3)
    |> softmax()
  end

  defn sigmoid(tensor) do
    1 / (1 + Nx.exp(-tensor))
  end

  defn softmax(tensor) do
    tensor = Nx.add(tensor, -Nx.reduce_max(tensor))
    tensor
    |> Nx.exp()
    |> Nx.divide( tensor |> Nx.exp() |> Nx.sum())
  end

  def acc(x,t,network) do
    {row, _} = Nx.shape(x)
    Enum.to_list(0..(row - 1))
    |> Nx.tensor()
    |> Nx.map(fn i ->
      predict(x[i],network)
      |> Nx.argmax()
      |> Nx.equal(t[i])
    end)
    |> Nx.sum
    |> Nx.divide(row)
  end

  def acc_batch(x,t,network) do
    x_b = x |> Nx.to_batched_list(100)
    t_b = t |> Nx.to_batched_list(100)
    {row, _} = x |> Nx.shape()
    batch = Enum.count(x_b)
    Enum.to_list(0..(batch-1))
    |> Nx.tensor()
    |> Nx.map(fn i ->
      predict(Enum.at(x_b,Nx.to_scalar(i)),network)
      |> Nx.argmax(axis: 1)
      |> Nx.equal(Enum.at(t_b,Nx.to_scalar(i)))
      |> Nx.sum()
    end)
    |> Nx.sum()
    |> Nx.divide(row)
  end
end
