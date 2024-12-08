  for (std::size_t row = rows.first; row < rows.second; ++row)
  {
    std::size_t ii = row - offset;
    std::vector<std::size_t> cols;

    std::vector<double> bij;
    Stiffness->getrow(row, cols, bij);
    
    // std::vector<double> c0ij;
    // std::vector<double> c1ij;

    // Cij_x0->getrow(row, cols, c0ij);
    // Cij_x1->getrow(row, cols, c1ij);
    
    std::vector<double> psiij(cols.size(), 0.0);

    double bij_udiff = 0.0;
    double bij_udiff_max = 0.0;

    for (std::size_t col = 0; col < cols.size(); ++col)
    {
      std::size_t jj = cols[col] - offset;
      if (jj + offset >= rows.first && jj + offset < rows.second)
      {
        if (ii != jj)
        {
            double diff_u = Fu_arr[jj] - Fu_arr[ii];

            bij_udiff += bij[col] * diff_u;
            bij_udiff_max += fabs(bij[col]) * fabs(diff_u);
        }
      }
    }

    double const alpha = fabs(bij_udiff) / std::max(bij_udiff_max, 1e-8);
    Falpha_arr[ii] = alpha * alpha;
  }
  Falpha_->vector()->set_local(Falpha_arr);
  Falpha_->vector()->apply("insert");
}