from data import SimulateData
from dcaf import RmSilenceGene, NotRmSilenceGene  


def  __main__():
    simulate_data = SimulateData(ncell=1000, ngene=5000)
    simulate_data.simulate_control()
    simulate_data.simulate_case()
    simulate_data.simulate_predicted(error_type='all', scale=0.05)
    low_thresh = 0.1

    rm_sil_gene = RmSilenceGene(
        simulate_data.simulated_control,
        simulate_data.simulated_case,
        simulate_data.simulated_predicted,
        low_thresh=low_thresh
    )

    not_rm_sil_gene = NotRmSilenceGene(
        simulate_data.simulated_control,
        simulate_data.simulated_case,
        simulate_data.simulated_predicted,
        low_thresh=low_thresh
    )
    not_rm_sil_gene.cal_log_fold_change()

    rm_sil_gene_measure = {'mse': rm_sil_gene.mse, 'mae': rm_sil_gene.mae}
    not_rm_sil_gene_measure = {'r2_lfc': not_rm_sil_gene.r2_lfc, 'mse_lfc': not_rm_sil_gene.mse_lfc}

    print('Measurement for remain silenced genes,', rm_sil_gene_measure,
          '\n',
          'Measurement for not remain silenced genes', not_rm_sil_gene_measure)


if __name__ == '__main__':
    __main__()
