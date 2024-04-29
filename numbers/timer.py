import time

from typing import Union
from datetime import datetime

class Timer:
  def __init__(self, name=None):
    if name == None:
      self.name = 'Timer'
    else:
      self.name = name
    
    self.started = None
    self.ended = None

  def start(self) -> None:
    self.started = time.time()
    print(f'{self.name} started... ({datetime.now().strftime("%H:%M:%S")})')

  def stop(self) -> None:
    self.ended = time.time()
    print(f'{self.name} done! | Time elapsed: {self.__get_formatted_time(self.ended - self.started)}\n')
    self._clear()

  def round(self) -> None:
    if self.started == None:
      return
    
    self.ended = time.time()
    print(f'Timer ({self.name}) currently running at: {self.__get_formatted_time(self.ended - self.started)}\n')

  def rename(self, name: str) -> None:
    self._clear()
    self.name = name

  def _clear(self):
    self.started, self.ended = None, None

  def __get_formatted_time(self, seconds: float) -> str:
    scaling_factor, unit = self.__get_time_scaling_factor(seconds), self.__get_time_unit(seconds)
    return f'{seconds * scaling_factor:.2f}{unit}'
  
  def __get_time_scaling_factor(self, seconds: str) -> Union[int, float]:
    if seconds < 1:
      return  1000
    elif seconds < 60:
      return 1
    elif seconds < 3600:
      return 1 / 60
    else:
      return 1 / 60

  def __get_time_unit(self, seconds: float):
    if seconds < 1:
      return 'ms'
    elif seconds < 60:
      return 's'
    else:
      return 'min'