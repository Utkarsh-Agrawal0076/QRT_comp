# QSEC Client


## Setup

* a python 3.11+ environment
* package dependencies: pandas and paramiko

```
# example environment setup with conda
conda create -n qsec python=3.11 -y
conda activate qsec
conda install -c conda-forge pandas paramiko ipyhton -y
```


## Usage

1. Generate a dataframe of target positions specifying at least `internal_code`, `currency` and `target_notional`
2. Use the function `perpare_targets_file` to generate a well formatted targets csv file.
3. use the function `upload_targets_file` to upload your targets csv to the SFTP server. 
You will need the SFTP host, your username and a private ssh key.


## License

Copyright &copy; 2024 [Qube Research and Technologies](https://www.qube-rt.com/ "Qube Research and Technologies homepage")

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
these files except in compliance with the License. You may obtain a copy of the
License at: http://www.apache.org/licenses/LICENSE-2.0, or from the LICENSE
file contained in this repository.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
