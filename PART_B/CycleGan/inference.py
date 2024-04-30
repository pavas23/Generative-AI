import torch
import torchvision.utils as vutils
from dataloaders import (
    men_no_glasses_loader,
    men_with_glasses_loader,
    women_with_glasses_loader,
)
from run import Generator
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"----------Creating Generators----------")

# Men without glasses to men with glasses
gen_A2B_men = Generator(3, 3).to(device)
gen_A2B_men.load_state_dict(torch.load("checkpoints/gen_A2B_men.pth"))
gen_A2B_men.eval()

print(f"----------Generator A2B men evaluated----------")

gen_B2A_men = Generator(3, 3).to(device)
gen_B2A_men.load_state_dict(torch.load("checkpoints/gen_B2A_men.pth"))
gen_B2A_men.eval()

print(f"----------Generator B2A men evaluated----------")

# Men with glasses to women with glasses
gen_A2B_women = Generator(3, 3).to(device)
gen_A2B_women.load_state_dict(torch.load("checkpoints/gen_A2B_women.pth"))
gen_A2B_women.eval()

print(f"----------Generator A2B women evaluated----------")

gen_B2A_women = Generator(3, 3).to(device)
gen_B2A_women.load_state_dict(torch.load("checkpoints/gen_B2A_women.pth"))
gen_B2A_women.eval()

print(f"----------Generator B2A women evaluated----------")


def test_cyclegan(
    model_A2B, model_B2A, dataloader_A, dataloader_B, output_folder, num_images=5
):
    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        real_A_images = []
        translated_images = []
        cycle_images = []

        for i, real_A in enumerate(dataloader_A):
            if i >= num_images:
                break

            real_A = real_A.to(device)
            translated = model_A2B(real_A)
            cycle = model_B2A(translated).permute(0, 3, 1, 2)

            real_A = real_A.permute(0, 3, 1, 2)
            translated = translated.permute(0, 3, 1, 2)
            real_A_images.append(vutils.make_grid(real_A))
            translated_images.append(vutils.make_grid(translated))
            cycle_images.append(vutils.make_grid(cycle))

        real_A_grid = torch.cat(real_A_images, dim=1)
        translated_grid = torch.cat(translated_images, dim=1)
        cycle_grid = torch.cat(cycle_images, dim=1)

        vutils.save_image(
            real_A_grid,
            os.path.join(output_folder, "A_input.png"),
        )
        vutils.save_image(
            translated_grid,
            os.path.join(output_folder, "A_translated.png"),
        )
        vutils.save_image(
            cycle_grid,
            os.path.join(output_folder, "A_cycle.png"),
        )

        print(f"Saved Images for A in {output_folder}")

        real_B_images = []
        translated_images = []
        cycle_images = []

        for i, real_B in enumerate(dataloader_B):
            if i >= num_images:
                break

            real_B = real_B.to(device)
            translated = model_B2A(real_B)
            cycle = model_A2B(translated).permute(0, 3, 1, 2)

            real_B = real_B.permute(0, 3, 1, 2)
            translated = translated.permute(0, 3, 1, 2)
            real_B_images.append(vutils.make_grid(real_B))
            translated_images.append(vutils.make_grid(translated))
            cycle_images.append(vutils.make_grid(cycle))

        real_B_grid = torch.cat(real_B_images, dim=1)
        translated_grid = torch.cat(translated_images, dim=1)
        cycle_grid = torch.cat(cycle_images, dim=1)

        vutils.save_image(
            real_B_grid,
            os.path.join(output_folder, "B_input.png"),
        )
        vutils.save_image(
            translated_grid,
            os.path.join(output_folder, "B_translated.png"),
        )
        vutils.save_image(
            cycle_grid,
            os.path.join(output_folder, "B_cycle.png"),
        )

        print(f"Saved Images for B in {output_folder}")


print(f"----------Testing CycleGAN----------")
print(f"----------Men Test----------")

if __name__ == "__main__":
    # test_cyclegan(gen_A2B_men, gen_B2A_men, men_no_glasses_loader, men_with_glasses_loader, "men")
    print(f"----------Women Testing----------")
    test_cyclegan(
        gen_A2B_women,
        gen_B2A_women,
        men_with_glasses_loader,
        women_with_glasses_loader,
        "women",
    )
    print(f"----------CycleGAN Testing Completed----------")
