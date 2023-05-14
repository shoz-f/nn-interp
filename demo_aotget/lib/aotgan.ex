defmodule AotGan do
  @width  512
  @height 512

  use NNInterp,
    model: "./model/aot_gan.pt",
    inputs: [f32: {1,3,@height,@width}, f32: {1,1,@height,@width}],
    outputs: [f32: {1,3,@height,@width}]

  def apply(img, mask) do
    # preprocess
    input0 = CImg.builder(img)
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:range, {-1.0, 1.0}}, :nchw, :bgr])

    input1 = CImg.builder(mask)
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:range, {0.0, 1.0}}, :nchw])

    # prediction
    session()
      |> NNInterp.set_input_tensor(0, input0)
      |> NNInterp.set_input_tensor(1, input1)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)
      |> CImg.from_binary(@width, @height, 1, 3, [{:range, {-1.0, 1.0}}, :nchw, :bgr])
      |> CImg.resize(img)
  end
end
