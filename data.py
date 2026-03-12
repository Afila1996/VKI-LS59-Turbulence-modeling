## Loading dataset
from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid

dataset = load_dataset("PLAID-datasets/VKI-LS59",split="all_samples")

dataset, problem = huggingface_dataset_to_plaid(dataset)

## Grid Construction 
GRID_SIZE = 64
BETA      = 10.0

def make_uniform_grid(grid_size):
    xs = np.linspace(0, 1, grid_size)
    ys = np.linspace(0, 1, grid_size)
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy

def interpolate_to_grid(nodes, values, grid_size=GRID_SIZE):

    xy_min = nodes.min(axis=0)
    xy_max = nodes.max(axis=0)
    nodes_norm = (nodes - xy_min) / (xy_max - xy_min + 1e-8)

    gx, gy = make_uniform_grid(grid_size)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])


    interp = griddata(nodes_norm, values, grid_points, method='cubic')
    mask   = np.isnan(interp)
    if mask.any():
        interp[mask] = griddata(nodes_norm, values, grid_points[mask], method='nearest')
    return interp.reshape(grid_size, grid_size)

def compute_chi(sdf_grid, beta=BETA):
    chi_sharp = (sdf_grid > 0).astype(np.float32)
    dist       = np.abs(sdf_grid)
    chi_smooth = np.tanh(beta * dist) * (chi_sharp - 0.5) + 0.5
    return chi_smooth.astype(np.float32)

## Dataset creation

X_all = []
Y_all = []
Chi_all = []
G = GRID_SIZE

for idx in ids_train:
    sample = dataset[idx]

    nodes    = sample.get_nodes(base_name="Base_2_2")        # (N, 2)
    sdf      = sample.get_field("sdf",  base_name="Base_2_2") # (N,)
    angle_in = sample.get_scalar("angle_in")
    mach_out = sample.get_scalar("mach_out")

    ro   = sample.get_field("ro",   base_name="Base_2_2")
    rou  = sample.get_field("rou",  base_name="Base_2_2")
    rov  = sample.get_field("rov",  base_name="Base_2_2")
    roe  = sample.get_field("roe",  base_name="Base_2_2")
    mach = sample.get_field("mach", base_name="Base_2_2")


    sdf_grid  = interpolate_to_grid(nodes, sdf)
    chi_grid  = compute_chi(sdf_grid)


    gx, gy = make_uniform_grid(G)
    angle_grid    = np.full((G, G), angle_in,  dtype=np.float32)
    mach_out_grid = np.full((G, G), mach_out,  dtype=np.float32)

    X = np.stack([gx, gy, sdf_grid, angle_grid, mach_out_grid], axis=-1)


    ro_g   = interpolate_to_grid(nodes, ro)
    rou_g  = interpolate_to_grid(nodes, rou)
    rov_g  = interpolate_to_grid(nodes, rov)
    roe_g  = interpolate_to_grid(nodes, roe)
    mach_g = interpolate_to_grid(nodes, mach)
    Y = np.stack([ro_g, rou_g, rov_g, roe_g, mach_g], axis=-1)

    X_all.append(X)
    Y_all.append(Y)
    Chi_all.append(chi_grid)

X_all   = np.stack(X_all)
Y_all   = np.stack(Y_all)
Chi_all = np.stack(Chi_all)

print(f"X_all shape : {X_all.shape}")
print(f"Y_all shape : {Y_all.shape}")
print(f"Chi_all shape : {Chi_all.shape}")

## normalisarion


from sklearn.model_selection import train_test_split


X_mean = X_all.mean(axis=(0, 1, 2))
X_std  = X_all.std( axis=(0, 1, 2))
Y_mean = Y_all.mean(axis=(0, 1, 2))
Y_std  = Y_all.std( axis=(0, 1, 2))



X_tr_raw, X_val_raw, Y_tr_raw, Y_val_raw, Chi_tr, Chi_val = train_test_split(
    X_all, Y_all, Chi_all, test_size=0.1, random_state=42)


X_tr  = (X_tr_raw  - X_mean) / (X_std  + 1e-8)
X_val = (X_val_raw - X_mean) / (X_std  + 1e-8)
Y_tr  = (Y_tr_raw  - Y_mean) / (Y_std  + 1e-8)
Y_val = (Y_val_raw - Y_mean) / (Y_std  + 1e-8)


print(f"  Train : {len(X_tr)}")
print(f"  Val   : {len(X_val)}")

## Train-test split

class CFDGridDataset(Dataset):
    def __init__(self, X, Y, Chi, rs_mass=None, rs_momx=None,
                 rs_momy=None, rs_ene=None):

        self.X   = torch.tensor(X,   dtype=torch.float32).permute(0,3,1,2)
        self.Y   = torch.tensor(Y,   dtype=torch.float32).permute(0,3,1,2)
        self.Chi = torch.tensor(Chi, dtype=torch.float32).unsqueeze(1)

        N, H, W = self.X.shape[0], self.X.shape[2], self.X.shape[3]
        zeros   = torch.zeros(N, H, W)
        self.rs_mass = rs_mass if rs_mass is not None else zeros
        self.rs_momx = rs_momx if rs_momx is not None else zeros
        self.rs_momy = rs_momy if rs_momy is not None else zeros
        self.rs_ene  = rs_ene  if rs_ene  is not None else zeros

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx], self.Chi[idx], self.rs_mass[idx],
                self.rs_momx[idx], self.rs_momy[idx], self.rs_ene[idx])


train_dataset = CFDGridDataset(X_tr,  Y_tr,  Chi_tr,
                               RS_tr_mass,  RS_tr_momx,
                               RS_tr_momy,  RS_tr_ene)
val_dataset   = CFDGridDataset(X_val, Y_val, Chi_val,
                               RS_val_mass, RS_val_momx,
                               RS_val_momy, RS_val_ene)

train_loader = DataLoader(train_dataset, batch_size=4,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=4,
                          shuffle=False, num_workers=0)


