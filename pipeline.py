# !/usr/bin/env python
"""

"""

import torch
import torch.utils.data
import torchio as tio
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from evaluation.metrics import (SoftNCutsLoss, ReconstructionLoss)
# from torchmetrics.functional import structural_similarity_index_measure
# from pytorch_msssim import ssim
from utils.results_analyser import *
from utils.vessel_utils import (load_model, load_model_with_amp, save_model, write_epoch_summary)

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


class Pipeline:

    def __init__(self, cmd_args, model, logger, dir_path, checkpoint_path, writer_training, writer_validating,
                 wandb=None):

        self.model = model
        self.logger = logger
        self.learning_rate = cmd_args.learning_rate
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cmd_args.learning_rate)
        self.num_epochs = cmd_args.num_epochs
        self.writer_training = writer_training
        self.writer_validating = writer_validating
        self.wandb = wandb
        self.CHECKPOINT_PATH = checkpoint_path
        self.DATASET_PATH = dir_path
        self.OUTPUT_PATH = cmd_args.output_path

        self.model_name = cmd_args.model_name
        self.clip_grads = cmd_args.clip_grads
        self.with_apex = cmd_args.apex
        self.num_classes = cmd_args.num_classes
        self.train_encoder_only = cmd_args.train_encoder_only

        # image input parameters
        self.patch_size = cmd_args.patch_size
        self.stride_depth = cmd_args.stride_depth
        self.stride_length = cmd_args.stride_length
        self.stride_width = cmd_args.stride_width
        self.samples_per_epoch = cmd_args.samples_per_epoch

        # execution configs
        self.batch_size = cmd_args.batch_size
        self.num_worker = cmd_args.num_worker

        # Losses
        self.s_ncut_loss_coeff = cmd_args.s_ncut_loss_coeff
        self.reconstr_loss_coeff = cmd_args.reconstr_loss_coeff

        # Following metrics can be used to evaluate
        self.radius = cmd_args.radius
        self.sigmaI = cmd_args.sigmaI
        self.sigmaX = cmd_args.sigmaX
        self.soft_ncut_loss = torch.nn.DataParallel(
            SoftNCutsLoss(radius=self.radius, sigma_i=self.sigmaI, sigma_x=self.sigmaX, patch_size=self.patch_size))
        self.soft_ncut_loss.cuda()
        # self.ssim = ssim  # structural_similarity_index_measure
        # self.ssim = structural_similarity_index_measure
        self.reconstruction_loss = ReconstructionLoss(recr_loss_model_path=cmd_args.recr_loss_model_path)
        self.reconstruction_loss.cuda()
        # self.dice = Dice()
        # self.focalTverskyLoss = FocalTverskyLoss()
        # self.iou = IOU()

        self.LOWEST_LOSS = float('inf')

        if self.with_apex:
            self.scaler = GradScaler()

        self.logger.info("Model Hyper Params: ")
        self.logger.info("\nLearning Rate: " + str(self.learning_rate))
        self.logger.info("\nNumber of Convolutional Blocks: " + str(cmd_args.num_conv))
        self.predictor_subject_name = cmd_args.predictor_subject_name

        if cmd_args.train:  # Only if training is to be performed
            training_set = Pipeline.create_tio_sub_ds(vol_path=self.DATASET_PATH + '/train/',
                                                      patch_size=self.patch_size,
                                                      samples_per_epoch=self.samples_per_epoch,
                                                      stride_length=self.stride_length, stride_width=self.stride_width,
                                                      stride_depth=self.stride_depth, num_worker=self.num_worker)
            self.train_loader = torch.utils.data.DataLoader(training_set, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=0)
            validation_set, num_subjects = Pipeline.create_tio_sub_ds(vol_path=self.DATASET_PATH + '/validate/',
                                                                      patch_size=self.patch_size,
                                                                      samples_per_epoch=self.samples_per_epoch,
                                                                      stride_length=self.stride_length,
                                                                      stride_width=self.stride_width,
                                                                      stride_depth=self.stride_depth,
                                                                      is_train=False, num_worker=self.num_worker)
            sampler = torch.utils.data.RandomSampler(data_source=validation_set, replacement=True,
                                                     num_samples=(self.samples_per_epoch // num_subjects) * 40)
            self.validate_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch_size,
                                                               shuffle=False, num_workers=0,
                                                               sampler=sampler)

    @staticmethod
    def create_tio_sub_ds(vol_path, patch_size, samples_per_epoch, stride_length, stride_width, stride_depth,
                          is_train=True, get_subjects_only=False, num_worker=0):

        vols = glob(vol_path + "*.nii") + glob(vol_path + "*.nii.gz")
        subjects = []
        for i in range(len(vols)):
            vol = vols[i]
            if "_mask" in vol:
                continue
            filename = os.path.basename(vol).split('.')[0]
            # subject = tio.Subject(
            #     img=tio.ScalarImage(vol),
            #     subjectname=filename,
            # )

            subject = tio.Subject(
                img=tio.ScalarImage(vol),
                sampling_map=tio.Image(vol.split('.')[0] + '_mask.nii.gz', type=tio.SAMPLING_MAP),
                subjectname=filename,
            )

            # vol_transforms = tio.ToCanonical(), tio.Resample(tio.ScalarImage(vol))
            # transform = tio.Compose(vol_transforms)
            # subject = transform(subject)
            subjects.append(subject)

        if get_subjects_only:
            return subjects

        if is_train:
            subjects_dataset = tio.SubjectsDataset(subjects)
            # sampler = tio.data.UniformSampler(patch_size)
            sampler = tio.data.WeightedSampler(patch_size, 'sampling_map')
            patches_queue = tio.Queue(
                subjects_dataset,
                max_length=(samples_per_epoch // len(subjects)) * 4,
                samples_per_volume=(samples_per_epoch // len(subjects)),
                sampler=sampler,
                num_workers=num_worker,
                start_background=True
            )
            return patches_queue
        else:
            overlap = np.subtract(patch_size, (stride_length, stride_width, stride_depth))
            grid_samplers = []
            for i in range(len(subjects)):
                grid_sampler = tio.inference.GridSampler(
                    subjects[i],
                    patch_size,
                    overlap,
                )
                grid_samplers.append(grid_sampler)
            return torch.utils.data.ConcatDataset(grid_samplers), len(grid_samplers)

    @staticmethod
    def normaliser(batch):
        for i in range(batch.shape[0]):
            if batch[i].max() > 0.0:
                batch[i] = batch[i] / batch[i].max()
        return batch

    def load(self, checkpoint_path=None, load_best=True):
        if checkpoint_path is None:
            checkpoint_path = self.CHECKPOINT_PATH

        if self.with_apex:
            self.model, self.optimizer, self.scaler = load_model_with_amp(self.model, self.optimizer, checkpoint_path,
                                                                          batch_index="best" if load_best else "last")
        else:
            self.model, self.optimizer = load_model(self.model, self.optimizer, checkpoint_path,
                                                    batch_index="best" if load_best else "last")

    def train(self):
        self.logger.debug("Training...")

        training_batch_index = 0
        for epoch in range(self.num_epochs):
            print("Train Epoch: " + str(epoch) + " of " + str(self.num_epochs))
            self.model.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            total_soft_ncut_loss = 0
            total_reconstr_loss = 0
            total_loss = 0
            batch_index = 0
            num_batches = 0

            for batch_index, patches_batch in enumerate(tqdm(self.train_loader)):

                local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                local_batch_mask = Pipeline.normaliser(patches_batch['sampling_map'][tio.DATA].float().cuda())
                local_batch_mask = local_batch_mask.expand((-1, self.num_classes, -1, -1, -1))
                # local_batch = torch.movedim(local_batch, -1, -3)
                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))

                # Clear gradients
                self.optimizer.zero_grad()
                with autocast(enabled=self.with_apex):
                    class_preds, reconstructed_patch = self.model(local_batch, local_batch_mask, ops="both")
                    soft_ncut_loss = self.soft_ncut_loss(local_batch, class_preds)
                    if torch.any(torch.isnan(soft_ncut_loss)):
                        self.logger.info("Found nan in soft_ncut_loss")
                        continue
                    soft_ncut_loss = self.s_ncut_loss_coeff * (soft_ncut_loss.sum() / local_batch.shape[0])
                    reconstructed_patch = torch.sigmoid(reconstructed_patch)
                    reconstruction_loss = self.reconstr_loss_coeff * self.reconstruction_loss(reconstructed_patch,
                                                                                              local_batch)

                # Update only encoder by backpropagating soft-n-cut loss
                if self.with_apex:
                    if type(soft_ncut_loss) is list:
                        for i in range(len(soft_ncut_loss)):
                            if i + 1 == len(soft_ncut_loss):  # final loss
                                # self.scaler.scale(soft_ncut_loss[i]).backward()
                                soft_ncut_loss[i].backward()
                            else:
                                # self.scaler.scale(soft_ncut_loss[i]).backward(retain_graph=True)
                                soft_ncut_loss[i].backward(retain_graph=True)
                        soft_ncut_loss = torch.sum(torch.stack(soft_ncut_loss))
                    else:
                        # self.scaler.scale(soft_ncut_loss).backward(retain_graph=True)
                        soft_ncut_loss.backward(retain_graph=True)
                    # soft_ncut_loss.backward()
                    # if self.clip_grads:
                    #     # self.scaler.unscale_(self.optimizer)
                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    # self.scaler.step(self.optimizer)
                    # self.scaler.update()
                    # self.optimizer.step()
                else:
                    soft_ncut_loss.backward(retain_graph=True)
                    # if self.clip_grads:
                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    # self.optimizer.step()

                # Update the WNet by backpropagating reconstruction_loss
                if self.with_apex:
                    if type(reconstruction_loss) is list:
                        for i in range(len(reconstruction_loss)):
                            if i + 1 == len(reconstruction_loss):  # final loss
                                # self.scaler.scale(reconstruction_loss[i]).backward()
                                reconstruction_loss[i].backward()
                            else:
                                # self.scaler.scale(reconstruction_loss[i]).backward(retain_graph=True)
                                reconstruction_loss[i].backward(retain_graph=True)
                        reconstruction_loss = torch.sum(torch.stack(reconstruction_loss))
                    else:
                        # self.scaler.scale(reconstruction_loss).backward()
                        reconstruction_loss.backward()
                    # reconstruction_loss.backward()
                    if self.clip_grads:
                        # self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    # self.scaler.step(self.optimizer)
                    # self.scaler.update()
                    self.optimizer.step()
                else:
                    reconstruction_loss.backward()
                    if self.clip_grads:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.optimizer.step()
                # try:
                # with autocast(enabled=self.with_apex):
                #     # Get the classification response map(normalized) and respective class assignments after argmax
                #     class_preds = self.model(local_batch, ops="enc")
                #     soft_ncut_loss = self.soft_ncut_loss(local_batch, class_preds)
                #     if torch.any(torch.isnan(soft_ncut_loss)):
                #         self.logger.info("Found nan in soft_ncut_loss")
                #         continue
                #     soft_ncut_loss = soft_ncut_loss.sum() / local_batch.shape[0]
                #
                # # Update only encoder by backpropagating soft-n-cut loss
                # if self.with_apex:
                #     # soft_ncut_loss.backward()
                #     self.scaler.scale(soft_ncut_loss).backward(retain_graph=True)
                #     if self.clip_grads:
                #         self.scaler.unscale_(self.optimizer)
                #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                #     self.scaler.step(self.optimizer)
                #     self.scaler.update()
                # else:
                #     soft_ncut_loss.backward(retain_graph=True)
                #     if self.clip_grads:
                #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                #     self.optimizer.step()
                #
                # reconstruction_loss = 0
                # if not str(self.train_encoder_only).lower() == "true":
                #     self.optimizer.zero_grad()
                #
                #     with autocast(enabled=self.with_apex):
                #         reconstructed_patch = self.model(local_batch, ops="dec")
                #         reconstructed_patch = torch.sigmoid(reconstructed_patch)
                #         reconstruction_loss = self.reconstruction_loss(reconstructed_patch, local_batch)
                #
                #     # Update the WNet by backpropagating reconstruction_loss
                #     if self.with_apex:
                #         # reconstruction_loss.backward()
                #         self.scaler.scale(reconstruction_loss).backward()
                #         if self.clip_grads:
                #             self.scaler.unscale_(self.optimizer)
                #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                #         self.scaler.step(self.optimizer)
                #         self.scaler.update()
                #     else:
                #         reconstruction_loss.backward()
                #         if self.clip_grads:
                #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                #         self.optimizer.step()

                loss = soft_ncut_loss + reconstruction_loss
                torch.cuda.empty_cache()

                # except Exception as error:
                #     self.logger.exception(error)
                #     sys.exit()

                self.logger.info("Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + " Training..." +
                                 "\n SoftNcutLoss: " + str(soft_ncut_loss) + " ReconstructionLoss: " +
                                 str(reconstruction_loss) + " total_loss: " + str(loss))

                # # Calculating gradients
                # if self.with_apex:
                #     if type(loss) is list:
                #         for i in range(len(loss)):
                #             if i + 1 == len(loss):  # final loss
                #                 self.scaler.scale(loss[i]).backward()
                #             else:
                #                 self.scaler.scale(loss[i]).backward(retain_graph=True)
                #         loss = torch.sum(torch.stack(loss))
                #     else:
                #         # self.scaler.scale(loss).backward()
                #         loss.backward()
                #
                #     if self.clip_grads:
                #         # self.scaler.unscale_(self.optimizer)
                #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                #         # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
                #     self.optimizer.step()
                #     # self.scaler.step(self.optimizer)
                #     # self.scaler.update()
                # else:
                #     loss.backward()
                #     if self.clip_grads:
                #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                #         # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
                #
                #     self.optimizer.step()

                # if training_batch_index % 50 == 0:  # Save best metric evaluation weights
                #     write_summary(writer=self.writer_training, index=training_batch_index,
                #                   similarity_loss=similarity_loss.detach().item(),
                #                   continuity_loss=avg_continuity_loss.detach().item(),
                #                   total_loss=loss.detach().item())
                training_batch_index += 1

                # Initialising the average loss metrics
                total_soft_ncut_loss += soft_ncut_loss.detach().item()
                try:
                    total_reconstr_loss += reconstruction_loss.detach().item()
                except Exception as detach_error:
                    total_reconstr_loss += reconstruction_loss
                total_loss += loss.detach().item()

                num_batches += 1

                # To avoid memory errors
                torch.cuda.empty_cache()

            # Calculate the average loss per batch in one epoch
            total_soft_ncut_loss /= (num_batches + 1.0)
            total_reconstr_loss /= (num_batches + 1.0)
            total_loss /= (num_batches + 1.0)

            # Print every epoch
            self.logger.info("Epoch:" + str(epoch) + " Average Training..." +
                             "\n SoftNcutLoss: " + str(total_soft_ncut_loss) + " ReconstructionLoss: " +
                             str(total_reconstr_loss) + " total_loss: " + str(total_loss))
            write_epoch_summary(writer=self.writer_training, index=epoch,
                                soft_ncut_loss=total_soft_ncut_loss,
                                reconstruction_loss=total_reconstr_loss,
                                total_loss=total_loss)
            if self.wandb is not None:
                self.wandb.log(
                    {"SoftNcutLoss_train": total_soft_ncut_loss, "ReconstructionLoss_train": total_reconstr_loss,
                     "total_loss_train": total_loss}, step=epoch)

            if self.with_apex:
                save_model(self.CHECKPOINT_PATH, {
                    'epoch_type': 'last',
                    'epoch': epoch,
                    # Let is always overwrite,we need just the last checkpoint and best checkpoint(saved after validate)
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'amp': self.scaler.state_dict()
                })
            else:
                save_model(self.CHECKPOINT_PATH, {
                    'epoch_type': 'last',
                    'epoch': epoch,
                    # Let is always overwrite,we need just the last checkpoint and best checkpoint(saved after validate)
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'amp': None
                })

            torch.cuda.empty_cache()  # to avoid memory errors
            self.validate(training_batch_index, epoch)
            torch.cuda.empty_cache()  # to avoid memory errors

        return self.model

    def validate(self, training_index, epoch):
        """
        Method to validate
        :param training_index: Epoch after which validation is performed(can be anything for test)
        :param epoch: Current training epoch
        :return:
        """
        self.logger.debug('Validating...')
        print("Validate Epoch: " + str(epoch) + " of " + str(self.num_epochs))

        total_soft_ncut_loss, total_reconstr_loss, total_loss = 0, 0, 0
        no_patches = 0
        self.model.eval()
        try:
            data_loader = self.validate_loader
        except Exception as error:
            validation_set, num_subjects = Pipeline.create_tio_sub_ds(vol_path=self.DATASET_PATH + '/validate/',
                                                                      patch_size=self.patch_size,
                                                                      samples_per_epoch=self.samples_per_epoch,
                                                                      stride_length=self.stride_length,
                                                                      stride_width=self.stride_width,
                                                                      stride_depth=self.stride_depth,
                                                                      is_train=False, num_worker=self.num_worker)
            sampler = torch.utils.data.RandomSampler(data_source=validation_set, replacement=True,
                                                     num_samples=(self.samples_per_epoch // num_subjects) * 40)
            data_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch_size,
                                                               shuffle=False, num_workers=0,
                                                               sampler=sampler)
        writer = self.writer_validating
        with torch.no_grad():
            for index, patches_batch in enumerate(tqdm(data_loader)):
                self.logger.info("loading" + str(index))

                local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                local_batch_mask = Pipeline.normaliser(patches_batch['sampling_map'][tio.DATA].float().cuda())
                local_batch_mask = local_batch_mask.expand((-1, self.num_classes, -1, -1, -1))
                # local_batch = torch.movedim(local_batch, -1, -3)

                try:
                    with autocast(enabled=self.with_apex):
                        # Get the classification response map(normalized) and respective class assignments after argmax
                        class_preds, reconstructed_patch = self.model(local_batch, local_batch_mask, ops="both")
                        soft_ncut_loss = self.soft_ncut_loss(local_batch, class_preds)
                        if torch.any(torch.isnan(soft_ncut_loss)):
                            self.logger.info("Found nan in soft_ncut_loss")
                            continue
                        soft_ncut_loss = self.s_ncut_loss_coeff * (soft_ncut_loss.sum() / local_batch.shape[0])
                        reconstructed_patch = torch.sigmoid(reconstructed_patch)
                        reconstruction_loss = self.reconstr_loss_coeff * self.reconstruction_loss(reconstructed_patch,
                                                                                                  local_batch)

                        if not str(self.train_encoder_only).lower() == "true":
                            loss = soft_ncut_loss + reconstruction_loss
                        else:
                            loss = soft_ncut_loss
                        torch.cuda.empty_cache()

                except Exception as error:
                    self.logger.exception(error)

                total_soft_ncut_loss += soft_ncut_loss.detach().item()
                total_reconstr_loss += reconstruction_loss.detach().item()
                total_loss += loss.detach().item()

                # Log validation losses
                self.logger.info("Batch_Index:" + str(index) + " Validation..." +
                                 "\n SoftNcutLoss: " + str(soft_ncut_loss) + " ReconstructionLoss: " +
                                 str(reconstruction_loss) + " total_loss: " + str(loss))
                no_patches += 1

        # Average the losses
        total_soft_ncut_loss = total_soft_ncut_loss / (no_patches + 1)
        total_reconstr_loss = total_reconstr_loss / (no_patches + 1)
        total_loss = total_loss / (no_patches + 1)

        process = ' Validating'
        self.logger.info("Epoch:" + str(training_index) + process + "..." +
                         "\n SoftNcutLoss:" + str(total_soft_ncut_loss) +
                         "\n ReconstructionLoss:" + str(total_reconstr_loss) +
                         "\n total_loss:" + str(total_loss))

        # write_summary(writer, training_index, similarity_loss=total_similarity_loss,
        #               continuity_loss=total_continuity_loss, total_loss=total_loss)
        write_epoch_summary(writer, epoch, soft_ncut_loss=total_soft_ncut_loss,
                            reconstruction_loss=total_reconstr_loss,
                            total_loss=total_loss)
        if self.wandb is not None:
            self.wandb.log({"SoftNcutLoss_val": total_soft_ncut_loss, "ReconstructionLoss_val": total_reconstr_loss,
                            "total_loss_val": total_loss}, step=epoch)

        if self.LOWEST_LOSS > total_loss:  # Save best metric evaluation weights
            self.LOWEST_LOSS = total_loss
            self.logger.info(
                'Best metric... @ epoch:' + str(training_index) + ' Current Lowest loss:' + str(self.LOWEST_LOSS))

            if self.with_apex:
                save_model(self.CHECKPOINT_PATH, {
                    'epoch_type': 'best',
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'amp': self.scaler.state_dict()})
            else:
                save_model(self.CHECKPOINT_PATH, {
                    'epoch_type': 'best',
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'amp': None})

    def test(self, test_logger, test_subjects=None, save_results=True):
        test_logger.debug('Testing...')
        result_root = os.path.join(self.OUTPUT_PATH, self.model_name, "results")
        os.makedirs(result_root, exist_ok=True)

        self.model.eval()

        with torch.no_grad():
            test_subject = test_subjects[0]
            subjectname = test_subject['subjectname']
            ip_vol = test_subject['img'][tio.DATA].float().cuda()
            ip_vol = torch.reshape(ip_vol[:, :, :, :144], (1, 1, 240, 240, 144))
            print("ip_vol shape: {}".format(ip_vol.shape))
            with autocast(enabled=self.with_apex):
                class_preds, reconstructed_patch = self.model(ip_vol, ops="both")
                reconstructed_patch = torch.sigmoid(reconstructed_patch)
                print("class_preds shape: {}".format(class_preds.shape))
                print("reconstructed_patch shape: {}".format(reconstructed_patch.shape))
                torch.save(class_preds, os.path.join(result_root, subjectname + "_WNET_v2_class_preds.pth"))
                torch.save(reconstructed_patch, os.path.join(result_root, subjectname + "_WNET_v2_recr.pth"))
                ignore, class_assignments = torch.max(class_preds, 1)
                print("class_assignments shape: {}".format(class_assignments.shape))
                class_assignments = class_assignments.cpu().squeeze().numpy().astype(np.uint16)
                reconstructed_patch = reconstructed_patch.cpu().squeeze().numpy().astype(np.uint16)
                save_nifti(class_assignments, os.path.join(result_root, subjectname + "_WNET_v2_seg_vol.nii.gz"))
                save_nifti(reconstructed_patch, os.path.join(result_root, subjectname + "_WNET_v2_recr.nii.gz"))

    def predict(self, image_path, label_path, predict_logger):
        image_name = os.path.basename(image_path).split('.')[0]

        sub_dict = {
            "img": tio.ScalarImage(image_path),
            "subjectname": image_name,
        }

        if bool(label_path):
            sub_dict["label"] = tio.LabelMap(label_path)

        subject = tio.Subject(**sub_dict)

        self.test(predict_logger, test_subjects=[subject], save_results=True)

    def extract_segmentation(self):
        print("Analysing predictions...")
        # result_root = os.path.join(self.OUTPUT_PATH, self.model_name, "results")
        # ignore, class_preds_max = torch.max(class_preds, 0)
        # class_preds_normalised = class_preds_max.numpy().astype(np.uint16)
        # save_nifti(class_preds_normalised, os.path.join(result_root, self.predictor_subject_name + "_WNET_seg.nii.gz"))

        # def cal_weight(self, raw_data, shape):

        radius = 4
        sigmaI = 10
        sigmaX = 4
        num_classes = 2
        patch = torch.ones(15, 1, 32, 32, 32)
        shape = patch.shape
        preds = torch.ones(15, num_classes, 32, 32, 32) / num_classes
        const_padding = torch.nn.ConstantPad3d(radius - 1, 0)
        padded_preds = const_padding(preds)
        # According to the weight formula, when Euclidean distance < r,the weight is 0, so reduce the dissim matrix size to radius-1 to save time and space.
        print("calculating weights.")
        dissim = torch.zeros(
            (shape[0], shape[1], shape[2], shape[3], shape[4], (radius - 1) * 2 + 1, (radius - 1) * 2 + 1,
             (radius - 1) * 2 + 1))
        padded_patch = torch.from_numpy(np.pad(patch, (
            (0, 0), (0, 0), (radius - 1, radius - 1), (radius - 1, radius - 1), (radius - 1, radius - 1)), 'constant'))
        for x in range(2 * (radius - 1) + 1):
            for y in range(2 * (radius - 1) + 1):
                for z in range(2 * (radius - 1) + 1):
                    dissim[:, :, :, :, :, x, y, z] = patch - padded_patch[:, :, x:shape[2] + x, y:shape[3] + y,
                                                             z:shape[4] + z]

        temp_dissim = torch.exp(-1 * torch.square(dissim) / sigmaI ** 2)
        dist = torch.zeros((2 * (radius - 1) + 1, 2 * (radius - 1) + 1, 2 * (radius - 1) + 1))
        for x in range(1 - radius, radius):
            for y in range(1 - radius, radius):
                for z in range(1 - radius, radius):
                    if x ** 2 + y ** 2 + z ** 2 < radius ** 2:
                        dist[x + radius - 1, y + radius - 1, z + radius - 1] = np.exp(
                            -(x ** 2 + y ** 2 + z ** 2) / sigmaX ** 2)

        print("weight calculated.")
        weight = torch.multiply(temp_dissim, dist)
        sum_weight = weight.sum(-1).sum(-1).sum(-1)

        # too many values to unpack
        cropped_seg = []
        for x in torch.arange((radius - 1) * 2 + 1, dtype=torch.long):
            width = []
            for y in torch.arange((radius - 1) * 2 + 1, dtype=torch.long):
                depth = []
                for z in torch.arange((radius - 1) * 2 + 1, dtype=torch.long):
                    depth.append(
                        padded_preds[:, :, x:x + preds.size()[2], y:y + preds.size()[3], z:z + preds.size()[4]].clone())
                width.append(torch.stack(depth, 5))
            cropped_seg.append(torch.stack(width, 5))
        cropped_seg = torch.stack(cropped_seg, 5)
        multi1 = cropped_seg.mul(weight)
        multi2 = multi1.sum(-1).sum(-1).sum(-1).mul(preds)
        multi3 = sum_weight.mul(preds)
        assocA = multi2.view(multi2.shape[0], multi2.shape[1], -1).sum(-1)
        assocV = multi3.view(multi3.shape[0], multi3.shape[1], -1).sum(-1)
        assoc = assocA.div(assocV).sum(-1)
        soft_ncut_loss = torch.add(-assoc, num_classes)
