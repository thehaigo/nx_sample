defmodule NxNn do
  @moduledoc """
  Documentation for `NxNn`.
  """

  @doc """
  Hello world.

  ## Examples

      iex> NxNn.hello()
      :world

  """
  def get_data() do
    x_test = Dataset.test_image() |> Nx.tensor() |> (& Nx.divide(&1, Nx.reduce_max(&1))).()
    t_test = Dataset.test_label() |> Nx.tensor()
    {x_test, t_test}
  end

  def init_network() do
    {w1,w2,w3,b1,b2,b3} = PklLoad.load("pkl/sample_weight.pkl")
    {
      Nx.tensor(w1),
      Nx.tensor(w2),
      Nx.tensor(w3),
      Nx.tensor(b1),
      Nx.tensor(b2),
      Nx.tensor(b3)
    }
  end

  def acc do
    {x, t} = get_data()
    IO.puts("Load Data")
    Benchee.run(%{
      "exla cpu" => fn -> MnistNx.acc(x, t, init_network()) end,
      "exla cpu batch" => fn -> MnistNx.acc_batch(x,t,init_network()) end,
      "nx" => fn -> Mnist.acc(x, t, init_network()) end,
      "nx batch" => fn ->  Mnist.acc(x,t,init_network()) end
    })
   end
end
