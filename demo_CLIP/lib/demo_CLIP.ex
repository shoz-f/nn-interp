defmodule ClipDemo do
  def run(img_path, text_list) do
    img = CImg.load(img_path)

    similarity = Clip.similarity(img, text_list)

    # softmax
    exp = Nx.exp(similarity)
    Nx.divide(exp, Nx.sum(exp))
  end
end