from main_divcolor import *

lab_dataset = LABDataset()
lab_dataset.load_data()
lab_dataset.make_dataset()
lab_dataset.make_dataloader(args.bs)

grey_dataset = GreyDataset()

mse = torch.nn.MSELoss(reduction="none").to(device)

divcolor = DivColor(args.d, args.pre)

divcolor.fit_vae(lab_dataset.train_loader, lab_dataset.val_loader, epochs=args.e, lr=args.lr_vae)

divcolor.make_latent_space(lab_dataset.train_set, "train")
divcolor.make_latent_space(lab_dataset.val_set, "val")

grey_dataset.load_data()
grey_dataset.make_dataset()
grey_dataset.make_dataloader(args.bs)

divcolor.fit_mdn(grey_dataset.train_loader, grey_dataset.val_loader, epochs=args.e, lr=args.lr_mdn)

# divcolor.colorize_one_image("C:/Users/mauricio/Pictures/IMG_20160710_212006.jpg", "C:/Users/mauricio/Pictures/grey.png")
# divcolor.colorize_images(grey_dataset.train_grey.cpu().numpy())