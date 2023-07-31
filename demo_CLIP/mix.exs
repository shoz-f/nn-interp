defmodule DemoCLIP.MixProject do
  use Mix.Project

  def project do
    [
      app: :demo_clip,
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
      mod: {Clip.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    System.put_env("NNINTERP", "libtorch")
    [
      {:nn_interp, path: ".."},
      {:nx, "~> 0.2.1"},
      {:npy, "~> 0.1.2"},
      {:cimg, "~> 0.1.21"}
    ]
  end
end
