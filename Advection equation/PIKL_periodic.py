# Allows to automatically switch from CPU to GPU
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float32)

def is_running_on_gpu():
  if torch.cuda.is_available():
    print("The algorithm is running on GPU.")
  else:
    print("The algorithm is not running on GPU.")


def Sob_formula(k1, k2, j1, j2, s, L):
    return torch.where(torch.logical_and(k1 == j1, k2 == j2), 1+ (k1**2 + k2**2)**s/(2*L)**(2*s), 0.)


def Sob_matrix(m, s, L, device):
  fourier_range = torch.arange(-m, m+1, device=device)
  k1, k2, j1, j2 = torch.meshgrid(fourier_range, fourier_range, fourier_range, fourier_range, indexing='ij')
  k1 = k1.flatten()
  k2 = k2.flatten()
  j1 = j1.flatten()
  j2 = j2.flatten()

  sob_values = Sob_formula(k1, k2, j1, j2, s, L)

  return sob_values.view((2*m+1)**2, (2*m+1)**2)

class DifferentialOperator:
    def __init__(self, coefficients=None):
        """
        Initialize the PDE.
        The keys are tuples representing the powers of d/dX and d/dY respectively.
        For example, {(2, 1): 3, (0, 0): -1} represents 3d^2/dX^2 d/dY - 1.
        """
        if coefficients is None:
            self.coefficients = {}
        else:
            self.coefficients = coefficients

    def __repr__(self):
        terms = []
        for (x_power, y_power), coefficient in sorted(self.coefficients.items(), reverse=True):
            if coefficient == 0:
                continue
            term = f"{coefficient}"
            if x_power != 0:
                term += f"*(d/dX)^{x_power}"
            if y_power != 0:
                term += f"*(d/dY)^{y_power}"
            terms.append(term)
        PDE = " + ".join(terms) if terms else "0"
        return "The PDE of your model is " + PDE + " = 0."

    def __add__(self, other):
        if isinstance(other, (int, float)):
            result = DifferentialOperator(self.coefficients.copy())
            if (0, 0) in result.coefficients:
                result.coefficients[(0, 0)] += other
            else:
                result.coefficients[(0, 0)] = other
            return result

        result = DifferentialOperator(self.coefficients.copy())
        for (x_power, y_power), coefficient in other.coefficients.items():
            if (x_power, y_power) in result.coefficients:
                result.coefficients[(x_power, y_power)] += coefficient
            else:
                result.coefficients[(x_power, y_power)] = coefficient
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        result = DifferentialOperator(self.coefficients.copy())
        for (x_power, y_power), coefficient in other.coefficients.items():
            if (x_power, y_power) in result.coefficients:
                result.coefficients[(x_power, y_power)] -= coefficient
            else:
                result.coefficients[(x_power, y_power)] = -coefficient
        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = DifferentialOperator()
            for (x_power, y_power), coefficient in self.coefficients.items():
                result.coefficients[(x_power, y_power)] = coefficient * other
            return result

        result = DifferentialOperator()
        for (x1, y1), c1 in self.coefficients.items():
            for (x2, y2), c2 in other.coefficients.items():
                power = (x1 + x2, y1 + y2)
                if power in result.coefficients:
                    result.coefficients[power] += c1 * c2
                else:
                    result.coefficients[power] = c1 * c2
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, exponent):
        if exponent == 0:
            return DifferentialOperator({(0, 0): 1})
        elif exponent < 0:
            raise ValueError("Exponent must be a non-negative integer")

        result = DifferentialOperator(self.coefficients.copy())
        for _ in range(1, exponent):
            result *= self
        return result

    def evaluate(self, x, y, L):
        total = 0
        geometry = 1j*torch.pi/2/L
        for (x_power, y_power), coefficient in self.coefficients.items():
            total += coefficient * (x ** x_power) * (y ** y_power) * (geometry **(x_power + y_power))
        return total

def Fourier_PDE(k1, k2, j1, j2, L, PDE):
  return torch.where(torch.logical_and(k1 == j1, k2 == j2), PDE.evaluate(k1,k2, L), 0.)

def PDE_matrix(m, L, PDE, device):
  fourier_range = torch.arange(-m, m+1, device=device)
  k1, k2, j1, j2 = torch.meshgrid(fourier_range, fourier_range, fourier_range, fourier_range, indexing='ij')
  k1 = k1.flatten()
  k2 = k2.flatten()
  j1 = j1.flatten()
  j2 = j2.flatten()

  PDE_values = Fourier_PDE(k1, k2, j1, j2, L, PDE)

  return PDE_values.view((2*m+1)**2, (2*m+1)**2)

def Omega_matrix(domain, m, device):
  if domain == "square":
    fourier_range = torch.arange(-m, m+1, device=device)
    k1, k2, j1, j2 = torch.meshgrid(fourier_range, fourier_range, fourier_range, fourier_range, indexing='ij')
    k1 = (k1-j1).flatten()
    k2 = (k2-j2).flatten()
    j1, j2 = None, None

    T_values =  torch.mul(torch.sinc(k1/2), torch.sinc(k2/2))/4

    return T_values.view((2*m+1)**2, (2*m+1)**2)
  elif domain == "disk":
    fourier_range = torch.arange(-m, m+1, device=device)
    k1, k2, j1, j2 = torch.meshgrid(fourier_range, fourier_range, fourier_range, fourier_range, indexing='ij')
    k1 = (k1-j1).flatten()
    k2 = (k2-j2).flatten()
    j1, j2 = None, None

    T_values = torch.where(torch.logical_or(k1!= 0, k2 != 0),
                           torch.special.bessel_j1(torch.pi/2*torch.sqrt(k1**2+k2**2))/4/torch.sqrt(k1**2+k2**2), torch.pi/16)
    return T_values.view((2*m+1)**2, (2*m+1)**2)
  elif domain == "square periodic in x":
    fourier_range = torch.arange(-m, m+1, device=device)
    k1, k2, j1, j2 = torch.meshgrid(fourier_range, fourier_range, fourier_range, fourier_range, indexing='ij')
    k1 = (k1-j1).flatten()
    k2 = (k2-j2).flatten()
    j1, j2 = None, None

    T_values = torch.sinc(k1/2)/2*torch.where(k2==0, 1,0) #torch.mul(torch.sinc(k1/2), torch.sinc(k2/2))/4

    return T_values.view((2*m+1)**2, (2*m+1)**2)

def phi_matrix(mat_x, mat_y, mat_j1, mat_j2, L):
  return torch.exp(torch.pi/L*(torch.mul(mat_x, mat_j1)+torch.mul(mat_y, mat_j2))*1j/2)

def M_mat(s, m, lambda_n, mu_n, L, domain, PDE, device):
  S = Sob_matrix(m, s, L, device)*(1.0+0*1j)
  P = PDE_matrix(m, L, PDE, device)*(1.0+0*1j)
  T = Omega_matrix(domain, m, device)*(1.0+0*1j)
  M = lambda_n * S + mu_n * torch.transpose(torch.conj_physical(P), 0, 1)@T@P
  return M

def RFF_fit(data_t, data_x, data_y, s, m, lambda_n, mu_n, L, domain, PDE, device):
  M = M_mat(s, m, lambda_n, mu_n, L, domain, PDE, device)
  return RFF(m, data_t, data_x, data_y, L,  M, device)

def RFF(m, data_t, data_x, data_y, L, M, device):
  l = len(data_x)

  mat_t = torch.tile(data_t, ((2*m+1)**2,1))
  mat_x = torch.tile(data_x, ((2*m+1)**2,1))

  fourier_range = torch.arange(-m, m+1, device=device)
  fourier_rangex = torch.arange(l, device=device)

  j1, j2,  k1 = torch.meshgrid( fourier_range, fourier_range, fourier_rangex, indexing='ij')
  j1 = j1.flatten().view((2*m+1)**2, l)
  j2 = j2.flatten().view((2*m+1)**2, l)

  phi_mat = phi_matrix(mat_t, mat_x, j1, j2, L)

  RFF_mat = phi_mat@torch.conj_physical(torch.transpose(phi_mat, 0, 1))
  data_y = data_y*(1.+0*1j)
  return torch.linalg.solve(RFF_mat+l*M, phi_mat@data_y)

def phi_z_mat(m, data_zt, data_zx, L, device):
  l2 = len(data_zx)
  mat_zt = torch.tile(data_zt, ((2*m+1)**2,1))
  mat_zx = torch.tile(data_zx, ((2*m+1)**2,1))
  fourier_range = torch.arange(-m, m+1, device=device)
  fourier_rangez = torch.arange(l2, device=device)
  jz1, jz2, k1, = torch.meshgrid(fourier_range, fourier_range, fourier_rangez,  indexing='ij')
  jz1 = jz1.flatten().view((2*m+1)**2, l2)
  jz2 = jz2.flatten().view((2*m+1)**2, l2)

  phi_z = phi_matrix(mat_zt, mat_zx, jz1, jz2, L)
  return phi_z


def RFF_estimate(regression_vect, data_zt, data_zx, s, m, n, lambda_n, mu_n, L, domain, PDE, device):
  phi_z = phi_z_mat(m, data_zt, data_zx, L, device)
  estimator = torch.transpose(torch.conj_physical(phi_z), 0,1)@regression_vect
  return estimator
