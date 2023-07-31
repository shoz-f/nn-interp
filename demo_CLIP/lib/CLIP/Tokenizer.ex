defmodule Clip.Tokenizer do
  use Agent

  defstruct bin2uni: %{}, uni2bin: %{}, encoder: %{}, decoder: %{}, bpedic: %{}, cache: %{}

  @doc """
  Returns list of utf-8 byte and a corresponding list of unicode strings.
  The reversible bpe codes work on unicode strings.
  This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
  When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
  This is a signficant percentage of your normal, say, 32K bpe vocab.
  To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
  And avoids mapping to whitespace/control characters the bpe code barfs on.
  """
  @u0000   Enum.concat([?!..?~, ?¡..?¬, ?®..?ÿ])
  @b2u     Enum.map(@u0000, &{&1, <<&1::utf8>>}) ++ Enum.with_index(Enum.reject(0..255, &(&1 in @u0000)), &{&1, <<&2+0x100::utf8>>})
  @unicode Enum.map(@b2u, &elem(&1, 1))

  @doc """
  Initialize SimpleTokenizer and register it as an Agent.
  """
  @default_bpe  "priv/bpe_simple_vocab_16e6.txt.gz"
  @bpe_size      (49152-256-2)

  def start_link(opts), do: init(opts)

  def init(_opts \\ []) do
    {:ok, bpe} = File.open(@default_bpe, [:read, :compressed, :utf8], fn file ->
      IO.stream(file, :line)
      |> Stream.drop(1)     # discard title line
      |> Stream.map(&String.split(&1))
      |> Enum.take(@bpe_size)
    end)

    encoder =
      into(%{}, Enum.with_index(@unicode))
      |> into(Enum.with_index(@unicode, fn k,v -> {k <> "</w>", 256+v} end))
      |> into(Enum.with_index(bpe, fn k,v -> {Enum.join(k), 512+v} end))
      |> Map.merge(%{"<|startoftext|>" => 512+@bpe_size, "<|endoftext|>" => 512+@bpe_size+1})

    Agent.start_link(fn ->
        %__MODULE__{
          bin2uni: into(%{}, @b2u),
          uni2bin: into(%{}, @b2u, fn {k,v} -> {v,k} end),
          encoder: encoder,
          decoder: Map.new(encoder, fn {k,v} -> {v,k} end),
          bpedic:  into(%{}, Enum.with_index(bpe, fn [first,second],v -> {[encoder[first],encoder[second]],512+v} end)),
          cache:   %{"<|startoftext|>" => [encoder["<|startoftext|>"]], "<|endoftext|>" => [encoder["<|endoftext|>"]]}
        }
      end, name: __MODULE__)
  end

  defp into(map, list, func \\ &(&1)) when is_map(map), do: Enum.into(list, map, func)

  @doc """
  Obtain dictionaies from the Agent.
  """
  def dictionaries() do
    Agent.get(__MODULE__, & &1)
  end

  @doc """
  Save the BPE token cache.
  """
  def save_cache(%__MODULE__{cache: cache}),
    do: Agent.update(__MODULE__, &%{&1| cache: cache})
  
  @doc """
  Reset the BPE token cache.
  """
  def reset_cache() do
    Agent.update(__MODULE__, &%{&1| cache: Map.take(&1.cache, ["<|startoftext|>","<|endoftext|>"])})
  end

  @doc """
  Encode text to tokens.
  """
  def encode(text, length \\ 0) do
    with dic <- dictionaries() do
      {tokens, dic} = whitespace_clean(text)
        |> split_into_words()
        |> Enum.map(&for <<c <- &1>>, into: "", do: dic.bin2uni[c])   # encode utf-8 bytes to unicode characters.
        |> to_bpe(dic)

      save_cache(dic)
      if length > 0, do: Enum.take(tokens, length), else: tokens
    end
  end

  def startoftext(), do: encode("<|startoftext|>")
  def endoftext(),   do: encode("<|endoftext|>")

  ## Compresses consecutive whitespace into a single space and converts letters to lowercase.
  defp whitespace_clean(text) do
    String.trim(text)
    |> String.replace(~r/\s+/, " ")
    |> String.downcase()
  end

  ## Split a sentence into a list of words based on the regular expression pattern.
  defp split_into_words(text) do
    Regex.scan(~r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"iu, text)
    |> List.flatten()
  end

  ## Converts Unicode words into BPE IDs.
  defp to_bpe(text, dic, result \\ [])
  defp to_bpe([], dic, result),
    do: {List.flatten(Enum.reverse(result)), dic}
  defp to_bpe([word|rest], %__MODULE__{cache: cache}=dic, result) do
    {bpe_ids, dic} =
      if is_map_key(cache, word), do: {cache[word], dic}, else: bpe(word, dic)

    to_bpe(rest, dic, [bpe_ids|result])
  end

  ## Returns the BPE ID corresponding to the word.
  defp bpe(word, %__MODULE__{encoder: encoder, bpedic: bpedic}=dic) do
    bpe_ids = String.split(word, "", trim: true)
      |> List.update_at(-1, &(&1 <> "</w>"))
      |> Enum.map(&(encoder[&1]))
      |> bpe_constructor(bpedic)

    {bpe_ids, Map.update!(dic, :cache, &Map.put(&1, word, bpe_ids))}
  end

  ## Create BPE ID.
  @inf 0x7fff_ffff
  defp bpe_constructor([x], _), do: [x]
  defp bpe_constructor(token, bpe_dic=%{}) do
    pair = Stream.chunk_every(token, 2, 1, :discard)
      |> Enum.map(&Map.get(bpe_dic, &1, @inf))

    case min_index(pair) do
      {_, @inf} ->
        token
      {index, id} ->
        {front, [_,_| rear]} = Enum.split(token, index)
        bpe_constructor(Enum.concat([front, [id], rear]), bpe_dic)
  end
  end

  ## Searches for the smallest value in the list and returns that value and its first index.
  defp min_index(list, index \\ 0, result \\ {-1, @inf})
  defp min_index([], _index, result),
    do: result
  defp min_index([x|rest], index, {min_index, min}),
    do: min_index(rest, index+1, if (x < min) do {index, x} else {min_index, min} end)

  @doc """
  Decode tokens to text.
  """
  def decode(tokens) do
    with dic <- dictionaries() do
      Enum.flat_map(tokens, &String.split(dic.decoder[&1], "", trim: true))
      |> Enum.reduce("", fn c,acc -> acc <> <<dic.uni2bin[c]>> end)
      |> String.replace("</w>", " ")
      |> String.trim()
    end
  end
end
