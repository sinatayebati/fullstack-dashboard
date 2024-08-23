import React from 'react';
import { 
    Box,
    useTheme
} from '@mui/material';
import { useGetCustomersQuery } from 'state/api';  
import Header from 'components/Header';
import { DataGrid } from '@mui/x-data-grid';

const Customers = () => {
    const theme = useTheme();
    const { data, isLoading } = useGetCustomersQuery();
    // console.log("🚀 ~ Customers ~ data:", data)

  return (
    <Box m="1.5rem 2.5rem">
        
    </Box>
  )
}

export default Customers
