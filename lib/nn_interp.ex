defmodule NNInterp do
  @moduledoc """
  Torch Script intepreter for Elixir.
  Deep Learning inference framework.
  """

  @basic_usage """
  ## Basic Usage
  You get the trained pytorch model and save it in a directory that your application can read.
  "your-app/priv" may be good choice.

  ```
  $ cp your-trained-model.pt ./priv
  ```

  Next, you will create a module that interfaces with the deep learning model.
  The module will need pre-processing and post-processing in addition to inference
  processing, as in the example following. NNInterp provides inference processing
  only.

  You put `use NNInterp` at the beginning of your module, specify the model path as an optional argument. In the inference
  section, you will put data input to the model (`NNInterp.set_input_tensor/3`), inference execution (`NNInterp.invoke/1`),
  and inference result retrieval (`NNInterp.get_output_tensor/2`).

  ```elixr:your_model.ex
  defmodule YourApp.YourModel do
    use NNInterp, model: "priv/your-trained-model.pt"

    def predict(data) do
      # preprocess
      #  to convert the data to be inferred to the input format of the model.
      input_bin = convert-float32-binaries(data)

      # inference
      #  typical I/O data for pytorch models is a serialized 32-bit float tensor.
      output_bin =
        __MODULE__
        |> NNInterp.set_input_tensor(0, input_bin)
        |> NNInterp.invoke()
        |> NNInterp.get_output_tensor(0)

      # postprocess
      #  add your post-processing here.
      #  you may need to reshape output_bin to tensor at first.
      tensor = output_bin
        |> Nx.from_binary({:f, 32})
        |> Nx.reshape({size-x, size-y, :auto})

      * your-postprocessing *
      ...
    end
  end
  ```
  """

  @timeout 300000
  @framework System.get_env("NNINTERP") || raise ArgumentError, "environment variable 'NNINTERP' must be set one of {tflite, onnxruntime, libtorch}."
  
  # the suffix expected for the model
  suffix = %{
    "tflite"      => ".tflite",
    "onnxruntime" => ".onnx",
    "libtorch"    => ".pt"
  }

  @model_suffix suffix[String.downcase(@framework)]

  # session record
  defstruct module: nil, inputs: [], outputs: []

  defmacro __using__(opts) do
    quote generated: true, location: :keep do
      use GenServer

      def start_link(opts) do
        GenServer.start_link(__MODULE__, opts, name: __MODULE__)
      end

      def init(opts) do
        executable = Application.app_dir(:nn_interp, "priv/nn_interp")
        opts = Keyword.merge(unquote(opts), opts)
        nn_model   = NNInterp.validate_model(Keyword.get(opts, :model), Keyword.get(opts, :url))
        nn_label   = Keyword.get(opts, :label, "none")
        nn_inputs  = Keyword.get(opts, :inputs, [])
        nn_outputs = Keyword.get(opts, :outputs, [])
        nn_opts    = Keyword.get(opts, :opts, "")

        port = Port.open({:spawn_executable, executable}, [
          {:args, String.split(nn_opts) ++ opt_tspecs("--inputs", nn_inputs) ++ opt_tspecs("--outputs", nn_outputs) ++ [nn_model, nn_label]},
          {:packet, 4},
          :binary
        ])

        {:ok, %{port: port, itempl: nn_inputs, otempl: nn_outputs}}
      end

      def session() do
        %NNInterp{module: __MODULE__}
      end

      def handle_call(cmd_line, _from, state) when is_binary(cmd_line) do
        Port.command(state.port, cmd_line)
        response = receive do
          {_, {:data, <<result::binary>>}} -> {:ok, result}
        after
          Keyword.get(unquote(opts), :timeout, 300000) -> {:timeout}
        end
        {:reply, response, state}
      end

      def handle_call({:itempl, index}, _from, %{itempl: template}=state) do
        {:reply, {:ok, Enum.at(template, index)}, state}
      end

      def handle_call({:otempl, index}, _from, %{otempl: template}=state) do
        {:reply, {:ok, Enum.at(template, index)}, state}
      end

      def terminate(_reason, state) do
        Port.close(state.port)
      end

      defp opt_tspecs(_, []), do: []
      defp opt_tspecs(opt_name, tspecs) do
        [opt_name, Enum.map(tspecs, &tspec2str/1) |> Enum.join(":")]
      end

      defp tspec2str({:skip, _}), do: ""
      defp tspec2str({dtype, shape}) do
        dtype = Atom.to_string(dtype)
        shape = Tuple.to_list(shape) |> Enum.map(fn :none->1; x->x end) |> Enum.join(",")
        "#{dtype},#{shape}"
      end
    end
  end


  @doc """
  Get name of backend NN framework.
  """
  def framework() do
    @framework
  end

  @doc """
  Ensure that the back-end framework is as expected.
  """
  def framework?(name) do
    unless String.downcase(name) == String.downcase(@framework),
      do: raise "Error: backend NN framework is \"#{@framework}\", not \"#{name}\"."
  end

  @doc """
  Ensure that the model matches the back-end framework.

  ## Parameters
    * model - path of model file
    * url - download site
  """
  def validate_model(nil, _), do: raise ArgumentError, "need a model file \"#{@model_suffix}\"."
  def validate_model(model, url) do
    validate_extname!(model)

	abs_path = Path.expand(model)
    unless File.exists?(abs_path) do
    	IO.puts("#{model}:")
        {:ok, _} = NNInterp.URL.download(url, Path.dirname(abs_path), Path.basename(abs_path))
    end
    model
  end

  defp validate_extname!(model) do
    actual_ext = Path.extname(model)
    unless actual_ext == @model_suffix,
      do: raise ArgumentError, "#{@framework} expects the model file \"#{@model_suffix}\" not \"#{actual_ext}\"."

    actual_ext
  end

  @doc """
  Get the propaty of the model.

  ## Parameters

    * mod - modules' names
  """
  def info(mod) do
    cmd = 0
    case GenServer.call(mod, <<cmd::little-integer-32>>, @timeout) do
      {:ok, result} ->  Poison.decode(result)
      any -> any
    end
  end

  @doc """
  Stop the interpreter.

  ## Parameters

    * mod - modules' names
  """
  def stop(mod) do
    GenServer.stop(mod)
  end

  @doc """
  Put a flat binary to the input tensor on the interpreter.

  ## Parameters

    * mod   - modules' names or session.
    * index - index of input tensor in the model
    * bin   - input data - flat binary, cf. serialized tensor
    * opts  - data conversion
  """
  def set_input_tensor(mod, index, bin, opts \\ [])

  def set_input_tensor(mod, index, bin, opts) when is_atom(mod) do
    cmd = 1
    case GenServer.call(mod, <<cmd::little-integer-32>> <> input_tensor(index, bin, opts), @timeout) do
      {:ok, result} ->  Poison.decode(result)
      any -> any
    end
    mod
  end

  def set_input_tensor(%NNInterp{inputs: inputs}=session, index, bin, opts) do
    %NNInterp{session | inputs: [input_tensor(index, bin, opts) | inputs]}
  end

  defp input_tensor(index, bin, opts) do
    dtype = case Keyword.get(opts, :dtype, "none") do
      "none" -> 0
      "<f4"  -> 1
      "<f2"  -> 2
    end
    {lo, hi} = Keyword.get(opts, :range, {0.0, 1.0})

    size = 16 + byte_size(bin)

    <<size::little-integer-32, index::little-integer-32, dtype::little-integer-32, lo::little-float-32, hi::little-float-32, bin::binary>>
  end

  @doc """
  Put flat binaries to the input tensors on the interpreter.

  ## Parameters

    * mod   - modules' names or session.
    * from  - first index of input tensor in the model
    * items - list of input data - flat binary, cf. serialized tensor
  """
  def set_input_tensors(mod, from, items) when is_list(items) do
    Enum.with_index(items, from)
    |> Enum.reduce(mod, fn {item, i}, mod -> set_input_tensor(mod, i, item) end)
  end
  
  @doc """
  Get the flat binary from the output tensor on the interpreter.

  ## Parameters

    * mod   - modules' names or session.
    * index - index of output tensor in the model
  """
  def get_output_tensor(mod, index, opts \\ [])

  def get_output_tensor(mod, index, _opts) when is_atom(mod) do
    cmd = 3
    case GenServer.call(mod, <<cmd::little-integer-32, index::little-integer-32>>, @timeout) do
      {:ok, result} -> result
      any -> any
    end
  end

  def get_output_tensor(%NNInterp{outputs: outputs}, index, _opts) do
    Enum.at(outputs, index)
  end

  @doc """
  Get list of the flat binary from the output tensoron the interpreter.

  ## Parameters

    * mod   - modules' names or session.
    * range - range of output tensor in the model
  """
  def get_output_tensors(mod, range) do
    for i <- range, do: get_output_tensor(mod, i)
  end

  @doc """
  Invoke prediction.

  Two modes are toggled depending on the type of input data.
  One is the stateful mode, in which input/output data are stored as model states.
  The other mode is stateless, where input/output data is stored in a session
  structure assigned to the application.

  ## Parameters

    * mod/session - modules name(stateful) or session structure(stateless).

  ## Examples.

    ```elixir
      output_bin = session()  # stateless mode
        |> NNInterp.set_input_tensor(0, input_bin)
        |> NNInterp.invoke()
        |> NNInterp.get_output_tensor(0)
    ```
  """
  def invoke(mod) when is_atom(mod) do
    cmd = 2
    case GenServer.call(mod, <<cmd::little-integer-32>>, @timeout) do
      {:ok, result} -> Poison.decode(result)
      any -> any
    end
    mod
  end

  def invoke(%NNInterp{module: mod, inputs: inputs}=session) do
    cmd   = 4
    count = Enum.count(inputs)
    data  = Enum.reduce(inputs, <<>>, fn x,acc -> acc <> x end)
    case GenServer.call(mod, <<cmd::little-integer-32, count::little-integer-32>> <> data, @timeout) do
      {:ok, <<count::little-integer-32, results::binary>>} ->
          if count > 0 do
              outputs = for <<size::little-integer-32, tensor::binary-size(size) <- results>> do tensor end
              %NNInterp{session | outputs: outputs}
          else
              "error: %{count}"
          end
      any -> any
    end
  end

  @deprecated "Use invoke/1 instead"
  def run(x), do: invoke(x)

  @doc """
  Execute post processing: nms.

  ## Parameters

    * mod             - modules' names
    * num_boxes       - number of candidate boxes
    * num_class       - number of category class
    * boxes           - binaries, serialized boxes tensor[`num_boxes`][4]; dtype: float32
    * scores          - binaries, serialized score tensor[`num_boxes`][`num_class`]; dtype: float32
    * opts
      * iou_threshold:   - IOU threshold
      * score_threshold: - score cutoff threshold
      * sigma:           - soft IOU parameter
      * boxrepr:         - type of box representation
         * :center  - center pos and width/height
         * :topleft - top-left pos and width/height
         * :corner  - top-left and bottom-right corner pos
  """

  def non_max_suppression_multi_class(mod, {num_boxes, num_class}, boxes, scores, opts \\ []) do
    box_repr = case Keyword.get(opts, :boxrepr, :center) do
      :center  -> 0
      :topleft -> 1
      :corner  -> 2
    end

    iou_threshold   = Keyword.get(opts, :iou_threshold, 0.5)
    score_threshold = Keyword.get(opts, :score_threshold, 0.25)
    sigma           = Keyword.get(opts, :sigma, 0.0)

    cmd = 5
    case GenServer.call(mod, <<cmd::little-integer-32, num_boxes::little-integer-32, box_repr::little-integer-32, num_class::little-integer-32, iou_threshold::little-float-32, score_threshold::little-float-32, sigma::little-float-32>> <> boxes <> scores, @timeout) do
      {:ok, nil} -> :notfind
      {:ok, result} -> Poison.decode(result)
      any -> any
    end
  end


  @doc """
  Adjust NMS result to aspect of the input image. (letterbox)

  ## Parameters:

    * nms_result - NMS result {:ok, result}
    * [rx, ry] - aspect ratio of the input image
  """
  def adjust2letterbox(nms_result, aspect \\ [1.0, 1.0])

  def adjust2letterbox({:ok, result}, [rx, ry]) do
    {
      :ok,
      Enum.reduce(Map.keys(result), result, fn key,map ->
        Map.update!(map, key, &Enum.map(&1, fn [score, x1, y1, x2, y2, index] ->
          x1 = if x1 < 0.0, do: 0.0, else: x1
          y1 = if y1 < 0.0, do: 0.0, else: y1
          x2 = if x2 > 1.0, do: 1.0, else: x2
          y2 = if y2 > 1.0, do: 1.0, else: y2
          [score, x1/rx, y1/ry, x2/rx, y2/ry, index]
        end))
      end)
    }
  end

  def adjust2letterbox(nms_result, _), do: nms_result
end
