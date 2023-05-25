defmodule DemoVGG16 do
  use NNInterp,
    model: "./model/vgg16-7.onnx",
    url: "https://github.com/shoz-f/nn-interp/releases/download/0.1.0/vgg16-7.onnx"

  @vgg16_shape {224, 224}
  @imagenet1000 (for item <- File.stream!("./imagenet1000.label") do
                    String.trim_trailing(item)
                  end)
                  |> Enum.with_index(&{&2, &1})
                  |> Enum.into(%{})

  def apply(img, top) do
    # preprocess
    bin = CImg.builder(img)
      |> CImg.resize(@vgg16_shape)
      |> CImg.to_binary([{:range, {-2.2, 2.7}}, :nchw])

    # prediction
    outputs = __MODULE__
      |> NNInterp.set_input_tensor(0, bin)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)
      |> Nx.from_binary({:f, 32}) |> Nx.reshape({1000})

    # postprocess
    outputs
    |> Nx.argsort(direction: :desc)
    |> Nx.slice([0], [top])
    |> Nx.to_flat_list()
    |> Enum.map(&@imagenet1000[&1])
  end
  
  def run() do
    unless File.exists?("lion.jpg"),
      do: NNInterp.URL.download("https://github.com/shoz-f/nn-interp/releases/download/0.1.0/lion.jpg")

    CImg.load("lion.jpg")
    |> __MODULE__.apply(3)
    |> IO.inspect()
    :ok
  end
end
