defmodule Clip do
  alias Clip.{Visual, Text}
  require Nx

  @logit_scale 100.0

  @doc """
  """
  def encode_image(image),
    do: Visual.encode(image)

  @doc """
  """
  def encode_text(text),
    do: Text.encode(text)

  def similarity(image, text) do
    image_feature = unit(Visual.encode(image))
    text_feature  = unit(Text.encode(text))

    Nx.dot(image_feature, Nx.transpose(text_feature))
    |> Nx.multiply(@logit_scale)
  end

  defp unit(t) when Nx.is_tensor(t),
    do: Nx.divide(t, Nx.reshape(Nx.LinAlg.norm(t, axes: [1]), {:auto, 1}))
end