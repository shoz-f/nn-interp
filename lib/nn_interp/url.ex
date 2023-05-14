defmodule NNInterp.URL do
  @doc """
  Download and process data from url.
  
  ## Parameters
    * url - download site url
    * func - function to process downloaded data
  """
  def download(url, func) when is_function(func) do
    IO.puts("Downloading \"#{url}\".")

    response = get!(url)

    IO.puts("...processing.")
    func.(response.body)
  end

  @doc """
  Download and save the file from url.
  
  ## Parameters
    * url - download site url
    * path - distination path of downloaded file
    * name - name for the downloaded file
  """
  def download(url, path \\ "./", name \\ nil)
  def download(nil, _, _), do: raise("error: need url of file.")
  def download(url, path, name) do
    IO.puts("Downloading from \"#{url}\".")

    response = get!(url)

    name = name || case attachment_filename(response.headers) do
      {:ok, name} -> name
      _ -> IO.puts("** 'noname.bin' was used due to lack of a valid file name **")
           "noname.bin"
    end

    File.mkdir_p(path)

    Path.join(path, name)
    |> save(response.body)
  end

  defp save(file, bin) do
    with :ok <- File.write(file, bin) do
      IO.puts("...finish.")
      {:ok, file}
    end
  end

  def get!(url) do
    http_opts = [
      ssl: [
        verify: :verify_peer,
        cacertfile: CAStore.file_path(),
        customize_hostname_check: [
          match_fun: :public_key.pkix_verify_hostname_match_fun(:https)
        ]
      ]
    ]

    case :httpc.request(:get, {url, []}, http_opts, stream: :self, sync: false) do
      {:ok, request_id} ->
        get_loop(request_id, [], &ProgressBar.render/2)
      {:error, reason} ->
        raise inspect(reason)
    end
  end
  
  defp get_loop(id, downloaded, progress \\ nil) do
    receive do
      {:http, reply_info} when elem(reply_info, 0) == id ->
        case Tuple.delete_at(reply_info, 0) do
          {:stream_start, headers} ->
            get_loop(id, downloaded, init_progress(progress, headers))
          {:stream, body} ->
            get_loop(id, [body|downloaded], render_progress(progress, body))
          {:stream_end, headers} ->
            %{headers: headers, body: IO.iodata_to_binary(Enum.reverse(downloaded))}
          {{status, _, _}} ->
            status
          any -> any
        end
    end
  end

  defp init_progress(render, headers) when is_function(render) do
    {_, length} = List.keyfind!(headers, 'content-length', 0)
    render = fn x -> render.(x, List.to_integer(length)) end

    {0, render}
  end
  defp init_progress(progress, _), do: progress

  defp render_progress({last_count, render}, bin) when is_function(render) do
    count = last_count + byte_size(bin)
    render.(count)

    {count, render}
  end
  defp render_progress(progress, _), do: progress

  defp attachment_filename(headers) do
    with {_, cd} <- List.keyfind(headers, 'content-disposition', 0),
         [[_, fname]] <- Regex.scan(~r/filename="?(.+)"?/, List.to_string(cd))
    do
      {:ok, fname}
    else
      _ -> :none
    end
  end
end
