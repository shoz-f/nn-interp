defmodule DemoVGG16.MixProject do
  use Mix.Project

  def project do
    [
      app: :demo_vgg16,
      version: "0.1.0",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {DemoVGG16.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    System.put_env("NNINTERP", "OnnxRuntime")
    [
      {:cimg, github: "shoz-f/cimg_ex"},
      {:nn_interp, path: "..", env: :test},
      {:nx, "~> 0.2.1"}
    ]
  end
end
