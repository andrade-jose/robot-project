# NOVO ARQUIVO: utils/metrics.py
import time
from functools import wraps

class PerformanceMetrics:
    def __init__(self):
        self.tempos_movimento = []
        self.sucessos = 0
        self.falhas = 0
    
    def tempo_movimento(self, func):
        """Decorator para medir tempo de movimentos"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                resultado = func(*args, **kwargs)
                duration = time.time() - start
                
                self.tempos_movimento.append(duration)
                if resultado:
                    self.sucessos += 1
                else:
                    self.falhas += 1
                    
                self.logger.info(f"Movimento levou {duration:.2f}s")
                return resultado
                
            except Exception as e:
                self.falhas += 1
                raise
                
        return wrapper
    
    def obter_estatisticas(self):
        if not self.tempos_movimento:
            return "Nenhum movimento registrado"
            
        return {
            'tempo_medio': np.mean(self.tempos_movimento),
            'tempo_max': max(self.tempos_movimento),
            'tempo_min': min(self.tempos_movimento),
            'taxa_sucesso': self.sucessos / (self.sucessos + self.falhas),
            'total_movimentos': len(self.tempos_movimento)
        }