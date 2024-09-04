import React from 'react';
import { ResponsivePie } from '@nivo/pie';
import { 
    Box, 
    Typography, 
    useTheme 
} from '@mui/material';
import { useGetSalesQuery } from 'state/api';

const BreakdownChart = () => {
    const { data, isLoading } = useGetSalesQuery();
    const theme = useTheme();

    if (!data || isLoading) return "Loading...";

    const colors = [
        theme.palette.secondary[500],
        theme.palette.secondary[300],
        theme.palette.secondary[300],
        theme.palette.secondary[500],
    ];
    const formattedData = Object.entries(data.salesByCategory).map(
        ([category, sales], i) => ({
            id: category,
            label: category,
            value: sales,
            color: colors[i]
        })
    );

  return (
    <div>
      
    </div>
  )
}

export default BreakdownChart
