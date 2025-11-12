# NCCL GPUDirect-TCPXO Guest Plugin

## Table of Contents

-   [About](#about)
-   [Prebuilt Docker Containers](#prebuilt-docker-containers)
-   [Supported Build Platform](#supported-build-platform)
-   [Supported Version](#supported-version)
-   [Prerequisites](#prerequisites)
-   [Building](#building)
-   [Build Troubleshooting](#build-troubleshooting)
-   [Deploying a Test Workload](#deploying-a-test-workload)
-   [Licenses](#licenses)
-   [Notice](#notice)

## About

This repository contains the guest components required to enable
[GPUDirect-TCPXO optimized NCCL communication](https://cloud.google.com/cluster-toolkit/docs/machine-learning/a3-mega-enable-gpudirect-tcpxo):

-   A network plugin for NCCL, designed to accelerate ML workload networking
    through a custom RDMA offload on our
    [IPUs](https://cloud.google.com/blog/products/compute/introducing-a3-supercomputers-with-nvidia-h100-gpus).
-   A persistent daemon, ***Receive Datapath Manager*** (RxDM), to register GPU
    buffers with the underlying network stack for offloading of data transfers
    from the CPU.

RxDM is typically run as a container. It communicates with the NCCL network
plugin using UNIX sockets. The network plugin is deployed along with your ML
workload.

## Prebuilt Docker Containers

Docker containers consisting of the artifacts of this repository are available
on Google Cloud Platform's (GCP) artifact repository for quick consumption.
These docker containers are built on the same code distributed here (exempting
the future libraries listed below). These docker containers will continue to be
provided as an alternative to setting up a build environment, for users who only
need to use GPUDirect-TCPXO as is.

You can find the latest docker container images, along with detailed release
notes, in our
[GPUDirect-TCPXO Release Notes](https://github.com/GoogleCloudPlatform/container-engine-accelerators/blob/master/gpudirect-tcpxo/README.md).
To enable `docker` to retrieve these images, see our [gcloud](#gcloud) section.

### Notes for Existing Users

-   These open-source containers are drop-in replacements for the previous
    closed-source releases.
-   The flags for RxDM have changed between the closed-source release and this
    open-source release. Starting with
    `us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/tcpgpudmarxd-dev:v1.0.18`,
    this is the new flag set for RxDM:

    ```
    --num_hops=2 --num_nics=8
    ```

    The flags that have been removed are `--uid=` and `--alsologtostderr`. If
    these these flags are provided, RxDM will print an error like so:

    ```
    ERROR: Accessing retired flag 'uid'
    ERROR: Accessing retired flag 'alsologtostderr'
    ```

    This is not a fatal error nor does it affect the workload in anyway.

### *Future Libraries*

In addition to RxDM and the network plugin, there are three more libraries that
may be used when running GPUDirect-TCPXO:

-   GPUViz
    -   Used for local collection of workload metrics.
-   The GCP NCCL Tuner Plugin
    -   This is GCP's implementation of NCCL's tuner plugin interface, tuned for
        performance on Google's networking infrastructure.
-   The Guest Config Checker
    -   This is a wrapper around the NCCL network plugin. On startup, it checks
        that the configuration options for GPUDirect-TCPXO are in line with our
        recommendations. It then loads the actual network plugin.

These libraries are also in the process of being open-sourced. For now, we are
providing them as closed-source binaries through a separate
[docker container](https://us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/nccl-plugin-gpudirect-tcpxo-precompiled-libs).

## Supported Build Platform

This project uses [bazel](https://bazel.build/about/intro) to build, and the
following platform provides the best experience:

-   Ubuntu 22.04

In the future, we want to make building easier for other platforms. For now, see
the [prerequisites](#prerequisites) and below and adjust accordingly for your
distribution.

## Supported Version

This version of GPUDirect-TCPXO has been tested against internal workloads for
performance and stability with the following NCCL and CUDA version:

CUDA | NCCL
---- | ---------
12.8 | v2.28.7-1

In addition to the above, we've verified that the following combinations are
buildable:

CUDA | NCCL
---- | ---------
12.2 | v2.21.5-1
12.4 | v2.21.5-1
12.4 | v2.23.4-1
12.8 | v2.19.3-1
12.8 | v2.21.5-1
12.8 | v2.23.4-1
12.8 | v2.26.5-1
12.8 | v2.27.5-1
12.8 | v2.28.3-1

## Prerequisites

At a glance, we need to install the following prerequisites for Ubuntu 22.04:

-   `git`
-   `wget`
-   `build-essential`
-   `python3.10` or higher
-   `python-is-python3`
-   `automake`
-   `libtool`
-   [Bazel](https://bazel.build/)
-   [Docker](https://www.docker.com/get-started/)
    -   Only needed if you intend to build the docker containers.
-   [`gcloud`](https://cloud.google.com/artifact-registry/docs/docker/authentication)
    -   Needed to retrieve the prebuilt docker images
-   [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
-   [NCCL (source)](https://github.com/NVIDIA/nccl)

The following instructions are for Ubuntu 22.04. To adapt these instructions to
other distributions, see our
[notes below for other distributions](#other-distributions).

### Packages

Back to Ubuntu 22.04, let's start with installing our package prerequisites:

```sh
sudo apt install -y git wget build-essential python3.10 python-is-python3 automake libtool
```

### Bazel

For Bazel, we recommend installing
[bazelisk](https://github.com/bazelbuild/bazelisk), which is a version manager
for Bazel. See the
[installation](https://github.com/bazelbuild/bazelisk?tab=readme-ov-file#installation)
section on bazelisk's GitHub page for the up-to-date instructions. The following
commands work at time of writing:

```sh
sudo apt install -y --no-install-recommends npm
sudo npm install -g @bazel/bazelisk
```

### Docker

For [Docker](https://docs.docker.com/engine/install/ubuntu/), we recommend using
its `apt` repository installation method. See the instructions in the
[Install using the `apt` repository](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)
section and then return here. Our build process expects the non-root user to be
able to run `docker`. The `apt` package for Docker sets up a `docker` group that
can run `docker` commands without `sudo`. After installing Docker, add the
non-root user to the group:

```sh
sudo usermod -aG docker $USER
```

Then log out and log back in so that the new group is active for this user. If
you don't want to log out/in, you can use the `newgrp` command:

```sh
newgrp docker
```

You should be able to run docker as the not-root user now:

```sh
my_non_root_user@my_host ~> docker run hello-world
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
17eec7bbc9d7: Pull complete
Digest: sha256:a0dfb02aac212703bfcb339d77d47ec32c8706ff250850ecc0e19c8737b18567
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/

```

### gcloud

To retrieve docker images from our GCP artifact repository, you must first
install authenticate with the GCP Artifact Registry.

See the instructions
[here](https://cloud.google.com/artifact-registry/docs/docker/authentication)
for configuring authentication. Make sure to include
[`us-docker.pkg.dev`](http://us-docker.pkg.dev/) as one of the registries.

### CUDA Toolkit

We only need the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit); we
do not need the driver nor kernel modules to build. This release is built and
tested against CUDA 12.8. See the
[CUDA download instructions](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)
to install the toolkit then return here. This link takes you to the instructions
for installing the CUDA Toolkit for Ubuntu 22.04 by adding NVIDIA's CUDA
repository as an `apt` source. There are other installation methods available on
that page, but the `apt` repository method is our recommendation.

Note: It is **important** that the major version of the CUDA Toolkit aligns with
the major version of the CUDA installation running on your deployment. If your
deployment is running CUDA `12.x`, you **must** build with CUDA Toolkit `12.x`.

### NCCL (Prerequisite)

NCCL should be retrieved from its [GitHub repo](https://github.com/NVIDIA/nccl).
The packaged releases (e.g., `.txz`, `.deb`, `.rpm`) do not contain the headers
needed to build a network plugin for NCCL (at time of writing). Therefore, we
need the NCCL source code itself.

For our workspace to build successfully, we require NCCL and CUDA to be
available at `/usr/local`. Here's the required directory structure:

```
/usr
\..
  /local
  \..
    /cuda-12.8
    /cuda -> cuda-12.8
    /nccl-v2.28.7-1
    /nccl -> nccl-v2.28.7-1
```

A typical CUDA Toolkit installation using NVIDIA's `.deb` packages will create a
`/usr/local` entry specific to that version, and make a symlink at
`/usr/local/cuda` through the
[Debian alternatives system](https://wiki.debian.org/DebianAlternatives). For
NCCL, we can leverage the same. The following instructions are for NCCL
`v2.28.7-1`, please adapt to the NCCL version you are using.

1.  Clone the repository to `/usr/local`:

    ```sh
    sudo git clone https://github.com/NVIDIA/nccl.git /usr/local/nccl-v2.28.7-1
    ```

2.  Checkout the desired version:

    ```sh
    cd /usr/local/nccl-v2.28.7-1
    sudo git checkout v2.28.7-1
    ```

3.  Add it to the alternatives system:

    ```sh
    sudo update-alternatives --install /usr/local/nccl nccl /usr/local/nccl-v2.28.7-1 22871
    ```

4.  Done! You can now use:

    ```sh
    sudo update-alternatives --config nccl
    ```

    To interactively configure the active NCCL version, or:

    ```sh
    sudo update-alternatives --set nccl /usr/local/nccl-v2.28.7-1
    ```

    To set it non-interactively.

    If this is the only version of NCCL you have installed, it will be already
    activated as the default link to `/usr/local/nccl`.

Other versions of NCCL exist as explicit git tags in the
[NCCL official repo](https://github.com/NVIDIA/nccl).

### Other Distributions

Here are some tips for setting up the prerequisites on non-Ubuntu distributions:

-   The packages installed through `apt` have analogs on many of the major
    distributions, or may share the same name. We call out some packages below
    that have special considerations:
    -   Python: our `bazel` workspace expects `python` in `PATH` to be a Python
        3.10 (or above) executable. If you are not sure how your distribution
        handles Python aliasing, consider using
        [`pyenv`](https://github.com/pyenv/pyenv) to ensure that you have the
        right version ofPpython installed and added to `PATH` correctly.
    -   C++ compiler: We require a C++20 compliant compiler to (e.g. GCC 10 and
        above). C++2a is insufficient. We require C++20 as indicated in the
        [`.bazelrc`](.bazelrc) file.
-   Bazelisk can still be installed through `npm` on other distributions, so
    install `npm` for your distribution and then go to the [Bazelisk](#bazel)
    section above.
    -   Installing `npm` through the
        [Node Version Manager](https://github.com/nvm-sh/nvm) also works, if
        your distribution doesn't provide `npm` or a recent version of `npm`.
-   Docker provides [instructions](https://docs.docker.com/engine/install/) for
    installing on many popular distributions. After installing, you should still
    add your non-root user to the `docker` group as we describe in the
    [Ubuntu Docker installation](#Docker).
    -   If you do not intend to build the docker containers, you may skip
        installing Docker.
-   The CUDA Toolkit page also provides
    [instructions](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64)
    for many popular distributions. After installing, ensure that
    `/usr/local/cuda` points to the CUDA Toolkit 12.8 installation.
-   NCCL: since we're just cloning a repository, the
    [installation instructions](#nccl-prerequisite) above still apply to other
    distributions with one caveat:

    -   If you're not on a platform that uses `update-alternatives`, you can
        simply create a symlink:

        ```sh
        sudo ln -s /usr/local/nccl-v2.28.7-1 /usr/local/nccl
        ```

-   One of our [workspace dependencies](MODULE.bazel) is
    [WebRTC](https://webrtc.org/). WebRTC itself is a massive library with many
    [dependencies](https://webrtc.github.io/webrtc-org/native-code/development/prerequisite-sw/).
    We've verified that these dependencies are correctly and successfully
    installed on Ubuntu 22.04. This may not work automatically for your chosen
    distribution. There are quite a lot of packages, so we won't enumerate them
    here. Please see
    [install-build-deps.py](https://source.chromium.org/chromium/chromium/src/+/main:build/install-build-deps.py)
    for the list of packages, particularly the functions `dev_list` and
    `lib_list`.

## Building

Make sure you've completed the prerequisites above for your distribution, before
proceeding with the build steps.

### Clone the Repo

```sh
git clone https://github.com/google/nccl-plugin-gpudirect-tcpxo
```

Make sure to use a non-root account.

### (Recommended) Align NCCL Versions

Before building, we recommend that the NCCL versions are aligned between the
TCPXO docker, your build host, and your deployment. If your deployment uses a
version of NCCL other than the one associated with this release, then make sure
to change the NCCL version at `/usr/local` on the build host and update the git
checkout for the TCPXO docker.

Note: We recommend that your deployment uses the same version of NCCL as this
release.

For example, if your deployment uses NCCL `v2.23.4-1`:

1.  Update [`prepare_source.sh`](prepare_source.sh). The `git checkout` line
    should read:

    ```sh
    git checkout -B github_nccl_2_23_4 68b542363f9a44cdaac480f51ebe0fc26de96139
    ```

    Note: The branch name is not interpreted in any way.

2.  Add and activate the corresponding NCCL version on the build host. See the
    [NCCL (Prerequisite)](#nccl-prerequisite) section for instructions.

### Using Provided Scripts

To simplify the build process, we've provided two scripts:

-   [`rxdm_build.sh`](rxdm_build.sh)
-   [`tcpxo_build.sh`](tcpxo_build.sh)

These scripts are meant to be run in sequence. The first builds WebRTC, the RxDM
binary, and its container. The second script builds the net plugin as well as
its container.

Using these two scripts is sufficient to:

-   Locally compile RxDM and the network plugin
-   Build the docker containers that will be ingested in your ML workloads

After setting up all the prerequisites (packages, CUDA, NCCL, Docker, Python,
etc), the build scripts should run successfully:

```sh
cd nccl-plugin-gpudirect-tcpxo
./rxdm_build.sh
./tcpxo_build.sh
```

The first script, `rxdm_build.sh`, builds WebRTC by invoking its build script.
On Debian-based platforms (e.g. Ubuntu), this build script checks for the
presence of WebRTC's build and runtime dependencies. If they are missing, it
will try to install them. This check may prompt you for `sudo` access to `apt
install` these missing packages.

If you run into issues with the RxDM or TCPXO build, check out the
[Build Troubleshooting](#build-troubleshooting) section further below.

In the subsequent sections, we'll cover the build steps encapsulated in these
two scripts in greater detail. These sections are for curious readers or those
who have needs beyond the build scripts.

Note: If all you want to do is build the containers, the above two scripts are
all you need to run.

#### Tagging the Docker Images

After building the docker images consider tagging them with a registry and
uploading them, to deploy with your workloads:

```sh
docker tag <IMAGE ID> my-docker.registry.dev/my-repository/my-image:my-tag
docker push my-docker.registry.dev/my-repository/my-image:my-tag
```

### Building WebRTC

We use SCTP as a communication protocol between the network plugin and the
offload server. SCTP is provided by WebRTC, so building WebRTC is our first
step. To make deployment simple, we build WebRTC as a static library so that the
linker includes it in the generated `libnccl-net.so`.

To build the WebRTC library, run:

```sh
bazel run webrtc:build_sctp -- $(realpath webrtc)
```

Like with NCCL, users can configure WebRTC to be built at a specific commit. The
script `bazel` invokes takes an optional commit hash as an argument. The default
commit hash is specified in [build_sctp.sh](webrtc/build_sctp.sh). Compatibility
with arbitrary WebRTC commits is not guaranteed. If you have a need to build
with a different version of WebRTC, you can specify the commit hash in the
`bazel` command:

```sh
bazel run webrtc:build_sctp -- $(realpath webrtc) [optional commit hash]
```

During the build process, you may be prompted for `sudo` access. This comes from
WebRTC's dependency check, where it is attempting to install its build/runtime
dependencies.

After a successful run, you will see additional files in the `webrtc` directory:

-   `libwebrtc.a` (Release)
-   `libwebrtc_debug.a` (Debug)
-   `include/` (headers)

This only needs to be done **ONCE per repo** and does not to be executed again,
even when rebuilding RxDM or the network plugin. This only needs to be re-run
when the WebRTC commit hash changes. Such hash changes will be communicated
clearly in our commits and release notes.

For existing commits of WebRTC please refer to its
[git src](https://webrtc.googlesource.com/src).

### Building RxDM

To build RxDM, build the target:

```sh
bazel build --compilation_mode=opt //buffer_mgmt_daemon:fastrak_gpumem_manager
```

After it successfully builds, `bazel` will print the location of the artifact
relative to the workspace root:

```sh
# ...
Target //buffer_mgmt_daemon:fastrak_gpumem_manager up-to-date:
  bazel-bin/buffer_mgmt_daemon/fastrak_gpumem_manager
# ...
```

### Building the NCCL plugin

To build the NCCL Fastrak Guest Plugin, build the target:

```sh
bazel build --compilation_mode=opt //tcpdirect_plugin/fastrak_offload:libnccl-net.so
```

After it successfully builds, `bazel` will print the location of the artifact
relative to the workspace root:

```sh
# ...
Target //tcpdirect_plugin/fastrak_offload:libnccl-net.so up-to-date:
  bazel-bin/tcpdirect_plugin/fastrak_offload/libnccl-net.so
# ...
```

### Manually Building the Docker Containers

The final step of the build scripts is building the docker containers for RxDM
and the network plugin.

#### Prepare the Sources

1.  Copy the generated RxDM binary and NCCL network plugin to a new folder in
    the repository root named `out`

    ```sh
    mkdir -p out
    cp bazel-bin/buffer_mgmt_daemon/fastrak_gpumem_manager bazel-bin/tcpdirect_plugin/fastrak_offload/libnccl-net.so out/
    ```

2.  Run the [`prepare_source.sh`](prepare_source.sh) script:

    ```sh
    ./prepare_source.sh
    ```

    -   This script clones the specific version of NCCL we've validated, where
        it will be copied into the tcpxo docker container.

#### Docker Build

Finally, build the desired docker container(s):

```sh
docker build -f rxdm.dockerfile .  # Optionally, tag the container -t my.repo.org/rxdm:v1
docker build -f tcpxo.dockerfile .
```

### Retrieving the Precompiled Libraries

The `tcpxo.dockerfile` automatically retrieves the precompiled libraries and
their supporting files from the precompiled library docker image. As discussed
in the [Future Libraries](#future-libraries) section, these libraries are:

-   GPUViz
-   Guest config checker
-   Tuner plugin

If you are not building the docker containers but still want the precompiled
libraries, we have included a script,
[retrieve_precompiled_libs.sh](retrieve_precompiled_libs.sh), to make retrieving
these libraries easy:

```
Usage: ./retrieve_precompiled_libs.sh [-i IMAGE_NAME] [-o OUTPUT_DIRECTORY]

Retrieves the precompiled libraries from the TCPXO plugin Docker image.

OPTIONS:
  -i    The name of the Docker image to use.
        (Default: us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/nccl-plugin-gpudirect-tcpxo-precompiled-libs:latest)
  -o    The directory where files will be copied.
        (Default: out/)
  -h    Show this help message.
```

This script will:

-   Download the docker image for the precompiled libraries
-   Spin up a docker container
-   Copy the libraries and their supporting files to a folder
    -   By default, it will copy to a folder named `out` and make it if it does
        not exist.
-   Delete the docker container

## Build Troubleshooting

Here are some issues that you may run into and how to resolve them:

##### rules_cuda

If you see an error like:

```
(01:40:52) ERROR: /home/my_user/.cache/bazel/_bazel_my_user/c757a90c1b6b527148a9070c3cadf01e/external/rules_cuda++toolchain+local_cuda/BUILD: no such target '@@rules_cuda++toolchain+local_cuda//:cuda': target 'cuda' not declared in package '' defined by /home/my_user/.cache/bazel/_bazel_my_user/c757a90c1b6b527148a9070c3cadf01e/external/rules_cuda++toolchain+local_cuda/BUILD
(01:40:52) ERROR: /home/my_user/nccl/nccl-plugin-fastrak/buffer_mgmt_daemon/BUILD:225:11: no such target '@@rules_cuda++toolchain+local_cuda//:cuda': target 'cuda' not declared in package '' defined by /home/my_user/.cache/bazel/_bazel_my_user/c757a90c1b6b527148a9070c3cadf01e/external/rules_cuda++toolchain+local_cuda/BUILD and referenced by '//buffer_mgmt_daemon:fastrak_gpumem_manager_lib'
(01:40:52) ERROR: /home/my_user/nccl/nccl-plugin-fastrak/buffer_mgmt_daemon/BUILD:225:11: no such target '@@rules_cuda++toolchain+local_cuda//:cuda_runtime': target 'cuda_runtime' not declared in package '' defined by /home/my_user/.cache/bazel/_bazel_my_user/c757a90c1b6b527148a9070c3cadf01e/external/rules_cuda++toolchain+local_cuda/BUILD and referenced by '//buffer_mgmt_daemon:fastrak_gpumem_manager_lib'
```

###### *Resolution*

Try building a simpler target first, like `bazel build
//buffer_mgmt_daemon:cuda_logging`

##### WebRTC

If you see an error like this:

```
Use --sandbox_debug to see verbose messages from the sandbox and retain the sandbox build root for debugging
In file included from dxs/client/dxs-client.cc:73:
./dxs/client/sctp-handler.h:27:10: fatal error: api/array_view.h: No such file or directory
   27 | #include "api/array_view.h"
      |          ^~~~~~~~~~~~~~~~~~
compilation terminated.
```

###### *Resolution*

Then the webrtc build step failed. One common failure is that the WebRTC commit
has been updated, meaning WebRTC needs to be rebuilt. Re-run `bazel run
webrtc:build_sctp -- $(realpath webrtc)`, and try your command again.

If this still doesn't work, then the WebRTC likely still failed. At this point,
you'll need to debug the error in the WebRTC `bazel run`.

##### GLIBC

If you see this error in your container logs (e.g. RxDM, TCPXO, etc):

```
/lib64/libc.so.6: version `GLIBC_2.36' not found
```

###### *Resolution*

Then your host might have a version of `libc` that's too new for the container.
Here are some options:

-   Try building on a host OS with an older version of `libc`.
-   Update the distribution version in the `Dockerfile` to a more recent version
    like Ubuntu 24:

    ```
    FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
    ```

## Deploying a Test Workload

Take a look at our cloud documentation for how to make use of GPUDirect-TCPXO in
your ML workloads:

<https://cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx>

We also have instructions for
[Slurm](https://slurm.schedmd.com/documentation.html):

<https://cloud.google.com/cluster-toolkit/docs/deploy/deploy-a3-mega-cluster>

These docs show you to make use of RxDM and the network plugin in a sample
workload, including how to configure arguments for RxDM and the environment
variables for the network plugin.

Note: These open-source containers are drop-in replacements for the previous
closed-source releases. The documentation from these resources are still
applicable and useful.

## Licenses

This project is under the [3-Clause BSD License](LICENSE). This project, among
other dependencies, makes use of NCCL and CUDA, which are under separate
licenses from NVIDIA.

### CUDA

CUDA is authored by NVIDIA.

The CUDA [BUILD](third_party/cuda_headers/BUILD) file, authored by us, makes the
CUDA headers available to our build system, but otherwise does not copy nor
modify the files. The CUDA headers themselves are under a custom EULA, with the
latest version at https://docs.nvidia.com/cuda/eula/index.html. This agreement
is also produced in the CUDA installation root, in `EULA.txt`.

### NCCL

NCCL is authored by NVIDIA.

The NCCL [BUILD](third_party/nccl/BUILD) file, authored by us, makes the NCCL
headers available to our build system, but otherwise does not copy nor modify
the files. The NCCL headers themselves are under the 3-Clause BSD License, with
the latest version at https://github.com/NVIDIA/nccl/blob/master/LICENSE.txt.
This license is also produced in the NCCL source code checkout root, in
`LICENSE.txt`.

## Notice

This is not an officially supported Google product. This project is not eligible
for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
