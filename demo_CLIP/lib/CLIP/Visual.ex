defmodule Clip.Visual do
  @width  224
  @height 224
  @feature_length 512

  use NNInterp,
    model: "./model/clip_visual.pt",
    url: "https://github.com/shoz-f/nn-interp/releases/download/0.1.1/clip_visual.pt",
    inputs: [f32: {1, 3, @width, @height}],
    outputs: [f32: {1, @feature_length}]

  def encode(imgs) when is_list(imgs),
    do: Nx.concatenate(Enum.map(imgs, &encode(&1)))

  def encode(img=%CImg{}) do
    # preprocess
    input0 = CImg.builder(img)
      |> CImg.resize({@width, @height}, :crop)
      |> CImg.to_binary([{:gauss, {{122.8, 68.5}, {116.7, 66.6}, {104.1, 70.3}}}, :nchw])

    # prediction
    output0 = __MODULE__
      |> NNInterp.set_input_tensor(0, input0)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)
      |> Nx.from_binary(:f32) |> Nx.reshape({1, @feature_length})

    # postprocess
    output0
  end
end
