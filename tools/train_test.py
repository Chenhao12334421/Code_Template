# pylint: skip-file
import os
import torch

def train(model, train_loader, test_loader, criterion, logger, args):
    """
    neccessary args:
    args.epochs,
    args.learning_rate,
    args.lr_decay_milestones,
    args.lr_decay_gamma=0.1,
    args.device,
    args.discription

    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
    )
    milestones = [int(ms) for ms in args.lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=args.lr_decay_gamma
    )
    best_acc = -1
    for epoch in range(args.epochs):
        model.train()
        for i, (data, target, actual_len) in enumerate(train_loader):
            optimizer.zero_grad()
            
            data, target, actual_len = data.to(args.device), target.to(args.device), actual_len.to(args.device)
            out = model(data, actual_len)
            loss = criterion(out, target)
            loss.backward()
            current_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            ## verbose information
            # if i % 10 == 0 and args.verbose:
            #     logger.info(
            #         f"Epoch {epoch}/{epochs}, iter {i}/{len(train_loader)}, loss={loss.item():.4f}, lr={current_lr:.5f}"
            #     )
        model.eval()
        acc, val_loss = test(model, test_loader, criterion, args)
        logger.info(
            f"Epoch {epoch}/{args.epochs}, Acc={acc:.4f}, Val Loss={val_loss:.4f}, lr={current_lr:.5f}"
        )
        if best_acc < acc:
            os.makedirs(args.output_dir, exist_ok=True)
            save_as = os.path.join(args.output_dir, f"{args.discription}.pth")
            torch.save(model.state_dict(), save_as)
            best_acc = acc
        scheduler.step()
    logger.info(f"Best Acc={best_acc:.4f}")


def test(model, test_loader, criterion, args):
    correct = 0
    total = 0
    loss = 0
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        for i, (data, target, actual_len) in enumerate(test_loader):
            data, target, actual_len = data.to(args.device), target.to(args.device), actual_len.to(args.device)
            out = model(data, actual_len)
            loss += criterion(out, target)
            pred = out.max(1)[1]
            correct += (pred == target).sum()
            total += len(target)
    return (correct / total).item(), (loss / total).item()
