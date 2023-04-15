defmodule DemoR18 do
  @width  224
  @height 224

  use NNInterp,
    model: "./data/resnet18.tflite",
    url: "https://github.com/shoz-f/nn-interp/releases/download/0.0.1/resnet18.tflite",
    inputs: [f4: {1, 3, @height, @width}],
    outputs: [f4: {1, 1000}]

  @imagenet1000 (for item <- File.stream!("./imagenet1000.label") do
                   String.trim_trailing(item)
                 end)
                |> Enum.with_index(&{&2, &1})
                |> Enum.into(%{})

  def apply(img, top \\ 1) do
    # preprocess
    bin = CImg.builder(img)
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:gauss, {{123.7, 58.4}, {116.3, 57.1}, {103.5, 57.4}}}, :nchw])

    # prediction
    outputs =
      __MODULE__
      |> NNInterp.set_input_tensor(0, bin)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)
      |> Nx.from_binary({:f, 32})
      |> Nx.reshape({1000})

    # postprocess
    exp = Nx.exp(outputs)

    # softmax
    Nx.divide(exp, Nx.sum(exp))
    |> Nx.argsort(direction: :desc)
    |> Nx.slice([0], [top])
    |> Nx.to_flat_list()
    |> Enum.map(&@imagenet1000[&1])
  end
  
  def run() do
    unless File.exists?("lion.jpg"),
      do: NNInterp.URL.download("https://github.com/shoz-f/nn-interp/releases/download/0.0.1/lion.jpg")

    CImg.load("lion.jpg")
    |> __MODULE__.apply(3)
    |> IO.inspect()
    :ok
  end
end
