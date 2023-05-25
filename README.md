# NNInterp
NNInterp is an integrated DNN interpreter for Elixir.
It is the successor to TflInterp and OnnxInterp, and allows you to choose a backend framework from "tflite",
"onnx-runtime" and "libtorch".

## Platform
I have confirmed it works in the following OS environment.

- Windows 10 with Visual C++ 2019
- WSL2/Ubuntu 20.04
- Linux Mint 20 "Ulyana"

## Requirements
cmake 3.13 or later is required.

Visual C++ 2019 for Windows.

## Installation
Add the following line to the list of dependencies in `mix.exs`.

```elixir
def deps do
  [
    {:nn_interp, "~> 0.1.0"}
  ]
end
```

## Basic Usage
First, select the back-end DNN framework. Set the environment variable NNINTERP to one of the following strings.

- tflite-cpu
- onnx-cpu
- libtorch-cpu

As a little trick, you can put the NNINTERP settings in mix.exs as shown below.

```elixir
def deps do
  System.put_env("NNINTERP", "onnx-cpu")
  [
    ...
  ]
end
```

Next, obtain the trained PyTorch model and save it in a directory accessible by your application. The "your-app/priv" 
directory could be a suitable choice.

```
$ cp your-trained-model.onnx ./priv
```

Create a module that interfaces with the deep learning model. This module will require pre-processing and 
post-processing functionality, in addition to the inference processing provided by NNInterp, as demonstrated in the 
following example.

At the beginning of your module, include the statement `use NNInterp` and specify the model path as an optional 
argument. In the inference section, you will need to set the data input for the model using  
`NNInterp.set_input_tensor/3`, execute the inference with `NNInterp.invoke/1`,  and retrieve the inference results via 
`NNInterp.get_output_tensor/2`.


```elixr:your_model.ex
defmodule YourApp.YourModel do
  use NNInterp,
    model: "priv/your-trained-model.onnx"

  def predict(data) do
    # preprocess
    #  to convert the data to be inferred to the input format of the model.
    input_bin = convert-float32-binaries(data)

    # inference
    #  typical I/O data for models is a serialized 32-bit float tensor.
    output_bin = session()
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

## Demo
A demo of ResNet18 running on "tflite", "onnx-runtime", or "libtorch" is available on the GitHub for this project. Please refer there.

Let's enjoy ;-)

## License
NNInterp is licensed under the Apache License Version 2.0.
