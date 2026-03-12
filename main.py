
LAMBDA_PHYS  = 1e-3
PHYS_WARMUP  = 50
PHYS_RAMP    = 50
EPOCHS       = 500
PATIENCE     = 150
GRAD_CLIP    = 1.0
SAVE_PATH    = 'best_model_vki_rans_physics.ckpt'


Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32).to(device)  # (5,)
Y_std_t  = torch.tensor(Y_std,  dtype=torch.float32).to(device)  # (5,)

def decode_prediction(pred_norm):

    return pred_norm * Y_std_t[None,:,None,None] \
                     + Y_mean_t[None,:,None,None]


torch.manual_seed(42)
model = iDAFNO2d(
    modes1=16, modes2=16, width=32, nlayer=4,
    inp_size=5, out_size=5
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5
)
criterion  = LpLoss(p=2, reduction='mean')

best_val_loss = float('inf')
early_stop    = 0
history = {'train_data':[], 'train_phys':[], 'train_total':[], 'val':[]}



for ep in range(EPOCHS):


    if ep < PHYS_WARMUP:
        phys_w = 0.0
    elif ep < PHYS_WARMUP + PHYS_RAMP:
        phys_w = LAMBDA_PHYS * (ep - PHYS_WARMUP) / PHYS_RAMP
    else:
        phys_w = LAMBDA_PHYS


    model.train()
    ep_data = ep_phys = ep_total = 0.0

    for x_b, y_b, chi_b, rs_mass, rs_momx, rs_momy, rs_ene in train_loader:

        x_b   = x_b.to(device)
        y_b   = y_b.to(device)
        chi_b = chi_b.to(device)

        optimizer.zero_grad()

        pred      = model(x_b, chi_b)
        loss_data = criterion(pred, y_b)

        if phys_w > 0.0:
            pred_phys = decode_prediction(pred)
            loss_phys, lm, lx, ly, le = rans_physics_loss(
                pred_phys,
                rs_mass, rs_momx, rs_momy, rs_ene
            )
            loss = loss_data + phys_w * loss_phys
        else:
            loss_phys = torch.tensor(0.0)
            loss      = loss_data

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        ep_data  += loss_data.item()
        ep_phys  += loss_phys.item()
        ep_total += loss.item()

    n        = len(train_loader)
    ep_data /= n;  ep_phys /= n;  ep_total /= n

    history['train_data' ].append(ep_data)
    history['train_phys' ].append(ep_phys)
    history['train_total'].append(ep_total)

    # VALIDATE
    model.eval()
    ep_val = 0.0
    with torch.no_grad():
        for x_b, y_b, chi_b, *_ in val_loader:
            pred    = model(x_b.to(device), chi_b.to(device))
            ep_val += criterion(pred, y_b.to(device)).item()
    ep_val /= len(val_loader)
    history['val'].append(ep_val)

    scheduler.step()


    if ep_val < best_val_loss:
        best_val_loss = ep_val
        early_stop    = 0
        torch.save({
            'epoch'      : ep,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'val_loss'   : best_val_loss,
        }, SAVE_PATH)
    else:
        early_stop += 1


    if ep % 50 == 0 or ep_val == best_val_loss:
        print(
            f"Ep {ep:4d} | "
            f"Total {ep_total:.5f} | "
            f"Data {ep_data:.5f} | "
            f"Phys {ep_phys:.5f} | "
            f"Val {ep_val:.5f} | "
            f"Best {best_val_loss:.5f} | "
            f"PhysW {phys_w:.1e} | "
            f"Pat {early_stop}/{PATIENCE}"
        )

    if early_stop >= PATIENCE:
        print(f"Early stopping at epoch {ep}")
        break

print(f"\nTraining complete — best val loss: {best_val_loss:.6f}")


