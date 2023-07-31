defmodule Clip.Text do
  @context_length 77
  @feature_length 512
  
  use NNInterp,
    model: "./model/clip_text.pt",
    url: "https://github.com/shoz-f/nn-interp/releases/download/0.1.1/clip_text.pt",
    inputs: [i32: {1, @context_length}],
    outputs: [f32: {1, @feature_length}]

  alias Clip.Tokenizer
  
  def encode(texts) when is_list(texts),
    do: Nx.concatenate(Enum.map(texts, &encode(&1)))

  def encode(text) when is_binary(text) do
    # preprocess
    ids = Tokenizer.startoftext()
          ++ Tokenizer.encode(text,  @context_length-2) # keep the length is less than equal to @context_length.
          ++ Tokenizer.endoftext()

    input0 = ids ++ List.duplicate(0, @context_length - Enum.count(ids))    # filling 0 after tokens.
      |> Enum.into("", &<<&1::little-integer-size(32)>>)

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
