defmodule NNInterp.URL do
  def download(url, path \\ "./", name \\ nil) do
    IO.puts("Downloading \"#{url}\".")
    with {:ok, res} <- HTTPoison.get(url, [], follow_redirect: true),
      bin <- res.body,
      {_, <<"attachment; filename=", fname::binary>>} <- List.keyfind(res.headers, "Content-Disposition", 0),
      :ok <- File.mkdir_p(path)
    do
      Path.join(path, name||fname)
      |> save(bin)
    end
  end

  defp save(file, bin) do
    with :ok <- File.write(file, bin) do
      IO.puts("...finish.")
      {:ok, file}
    end
  end
end
