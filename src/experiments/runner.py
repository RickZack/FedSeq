import asyncio
import os
from datetime import datetime
from numpy import random
import subprocess
from typing import Callable, Iterable, List, Dict, Any, Optional
from omegaconf import DictConfig
from src.experiments.fedmodel import Param
from src.experiments import FedModel
from abc import ABC, abstractmethod


class Runner(ABC):
    def __init__(self, seed: int):
        self.seed = seed

    @abstractmethod
    def run(self, python_file: str, model: FedModel, name: str, times: int = 1, *args, **kwargs):
        pass

    def wait_all(self):
        pass

    @staticmethod
    def run_command(python_file: str, model: FedModel) -> List[str]:
        cmd_line_args = [p.as_cmd_arg() for p in model.params.values()]
        cmd_line_command = ['python3', python_file, *cmd_line_args]
        return cmd_line_command

    @staticmethod
    def aggregation_command(experiment_dir) -> List[str]:
        cmd_line_command = ['python3', "aggregate_fits.py", f'folder={experiment_dir}']
        return cmd_line_command

    @abstractmethod
    def run_aggregation(self, name: str, experiment_dir: str, dependencies: Iterable[str]):
        pass


class LocalRunner(Runner):
    def __init__(self, seed: int = 2021):
        super().__init__(seed)
        self._returnCodes: List[int] = []

    def run(self, python_file: str, model: FedModel, name: str, times: int = 1, *args, **kwargs):
        loop = asyncio.get_event_loop()
        random.seed(self.seed)
        seeds = random.randint(0, 1000000, times)
        for i in range(times):
            model.set_param("output_suffix", Param("output_suffix", f"_iteration{i + 1}"))
            model.set_param("seed", Param("seed", seeds[i]))
            cmd_line_command = Runner.run_command(python_file, model)
            loop.run_until_complete(self.__async_run(cmd_line_command))
        model.set_success(self._returnCodes[-1] == 0)
        self.run_aggregation(name, model.params["savedir"].value)

    async def __async_run(self, cmd_line_command):
        proc = await asyncio.create_subprocess_exec(*cmd_line_command)
        self._returnCodes.append(await proc.wait())

    def run_aggregation(self, name: str, experiment_dir: str, *args, **kwargs):
        cmd_line_command = Runner.aggregation_command(experiment_dir)
        cp: subprocess.CompletedProcess = subprocess.run(cmd_line_command, stdout=subprocess.PIPE)
        if cp.returncode != 0:
            print(f"WARNING: aggregation for {name} exited with exit code {cp.returncode}")


class ParallelLocalRunner(LocalRunner):
    def __init__(self, seed: int = 2021):
        super().__init__(seed)
        self.__processes: List[asyncio.subprocess.Process] = []
        self.__callbacks: List[Callable[[bool], Any]] = []

    def run(self, python_file: str, model: FedModel, name: str, times: int = 1, *args, **kwargs):
        loop = asyncio.get_event_loop()
        experiment_dir = model.params["savedir"]
        callback = lambda outcome, m=model, obj=self: [m.set_success(outcome),
                                                       obj.run_aggregation(name, experiment_dir)]
        random.seed(self.seed)
        seeds = random.randint(0, 1000000, times)
        for i in range(times):
            model.update("output_suffix", Param("output_suffix", f"_iteration{i + 1}"))
            model.update("seed", Param("seed", seeds[i]))
            cmd_line_command = Runner.run_command(python_file, model)
            loop.run_until_complete(self.__async_run(cmd_line_command))
        self.__callbacks.append(callback)

    async def __async_run(self, cmd_line_command):
        proc = await asyncio.create_subprocess_exec(*cmd_line_command)
        self.__processes.append(proc)

    def wait_all(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.__async_wait_all())

    async def __async_wait_all(self):
        for proc, callback in zip(self.__processes, self.__callbacks):
            ret_code = await proc.wait()
            self._returnCodes.append(ret_code)
            callback(ret_code == 0)


class SlurmException(Exception):
    pass


def sbatch(script_path: str):
    cp: subprocess.CompletedProcess = subprocess.run(["sbatch", script_path],
                                                     stdout=subprocess.PIPE)
    if cp.returncode != 0:
        raise SlurmException(f"Slurm returned with exit code {cp.returncode}")
    return cp.stdout.decode().strip("\n").split(" ")[-1]  # job code from sbatch output


class SlurmRunner(Runner):
    def __init__(self, seed: int, default_params: DictConfig, scripts_dir: Optional[str] = None, prep_cmd: str ='',
                 defaults: Optional[Dict[str, str]] = None, run_sbatch: bool = True):
        super().__init__(seed)
        defaults = defaults or {}
        self.__jobs: Dict[str, str] = dict()
        self.default_params = default_params
        self.__scripts_dir = scripts_dir
        self.__prep_cmd = prep_cmd
        self.__run_sbatch = run_sbatch
        if not scripts_dir:
            self.__scripts_dir = os.path.join(os.getcwd(), "output", f"slurmScripts{datetime.now()}")
        os.makedirs(self.__scripts_dir, exist_ok=True)
        self.default_options = {"--gres": "gpu:1", "--cpus-per-task": "1", "--partition": "cuda"}
        self.default_options.update(defaults)

    def __run_options(self, run_name: str, options: Dict[str, str]):
        run_defaults = self.default_options.copy()
        run_defaults.update({"--job-name": run_name, **options})
        return run_defaults

    def run(self, python_file: str, model: FedModel, name: str, times: int = 1,
            options: Optional[Dict[str, str]] = None, *args, **kwargs):
        options = options or {}
        random.seed(self.seed)
        seeds = random.randint(0, 1000000, times)
        dependencies = []
        for i in range(times):
            new_name = f"{name}_iteration{i + 1}"
            run_options = self.__run_options(new_name, options)
            model.set_param("output_suffix", Param("output_suffix", f"_iteration{i + 1}"))
            model.set_param("seed", Param("seed", seeds[i]))
            cmd_line_command = Runner.run_command(python_file, model)
            SlurmRunner.make_script(cmd_line_command, self.__scripts_dir, new_name, run_options, self.__prep_cmd)
            if self.__run_sbatch:
                self.__jobs[new_name] = sbatch(os.path.join(self.__scripts_dir, f"{new_name}.sh"))
                dependencies.append(self.__jobs[new_name])
        if times > 1 and self.__run_sbatch:
            self.run_aggregation(name, model.params["savedir"].value, dependencies)

    def run_aggregation(self, name, experiment_dir, dependencies):
        cmd_line_command = Runner.aggregation_command(experiment_dir)
        job_name = f"{name}_aggregation"
        job_list = ':'.join(dependencies)
        run_options = {"--dependency": f"afterok:{job_list}", "--job-name": job_name}
        SlurmRunner.make_script(cmd_line_command, self.__scripts_dir, job_name, run_options, self.__prep_cmd)
        self.__jobs[job_name] = sbatch(os.path.join(self.__scripts_dir, f"{job_name}.sh"))

    @staticmethod
    def make_script(cmd_line_command: List[str], where: str, name: str, options: Dict[str, str], prep_cmd: str):
        with open(os.path.join(where, f"{name}.sh"), "wt") as s:
            s.write("#!/bin/bash\n")
            [s.write(f"#SBATCH {op_name}={op_value}\n") for op_name, op_value in options.items()]
            s.write(f"cd {os.getcwd()}\n")
            s.write(prep_cmd)
            s.write(' \\\n\t'.join(cmd_line_command))
        s.close()

    def wait_all(self):
        with open(os.path.join(self.__scripts_dir, "jobs.txt"), "wt") as f:
            [f.write(f"{job_name}:{job_code}\n") for job_name, job_code in self.__jobs.items()]
        f.close()
