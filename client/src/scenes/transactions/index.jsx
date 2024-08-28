import React, { useState, useEffect } from 'react';
import { DataGrid } from '@mui/x-data-grid';
import { useGetTransactionsQuery } from 'state/api';
import Header from 'components/Header';
import { Box, useTheme } from "@mui/material";
import DataGridCustomToolbar from 'components/DataGridCustomToolbar';

const Transactions = () => {
  const theme = useTheme();

  // values to be sent to the backend
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(20);
  const [sort, setSort] = useState({});
  const [search, setSearch] = useState("");

  const [searchInput, setSearchInput] = useState("");
  
  const { data, isLoading, refetch } = useGetTransactionsQuery({
    page: page + 1,
    pageSize,
    sort: JSON.stringify(sort),
    search,
  });

  useEffect(() => {
    console.log("Page changed to:", page);
    refetch();
  }, [page, pageSize, sort, search, refetch]);

  console.log("Transaction ~ data:", data);
  console.log("Current page:", page);
  console.log("Current pageSize:", pageSize);

  const columns = [
    {
      field: "_id",
      headerName: "ID",
      flex: 1,
    },
    {
      field: "userId",
      headerName: "User ID",
      flex: 1,
    },
    {
      field: "createdAt",
      headerName: "CreatedAt",
      flex: 1,
    },
    {
      field: "products",
      headerName: "# of Products",
      flex: 0.5,
      sortable: false,
      renderCell: (params) => params.value.length,
    },
    {
      field: "cost",
      headerName: "Cost",
      flex: 1,
      renderCell: (params) => `$${Number(params.value).toFixed(2)}`,
    },
  ];

  const handlePageChange = (newPage) => {
    console.log("Page change requested:", newPage);
    setPage(newPage);
  };

  const handlePageSizeChange = (newPageSize) => {
    console.log("Page size change requested:", newPageSize);
    setPageSize(newPageSize);
  };

  return (
    <Box m="1.5rem 2.5rem">
        <Header title="TRANSACTIONS" subTitle="Entire list of transactions" />
        <Box 
            height="80vh"
            sx={{
              "& .MuiDataGrid-root": {
                border: "none",
              },
              "& .MuiDataGrid-cell": {
                borderBottom: "none",
              },
              "& .MuiDataGrid-columnHeaders": {
                backgroundColor: theme.palette.background.alt,
                color: theme.palette.secondary[100],
                borderBottom: "none",
              },
              "& .MuiDataGrid-virtualScroller": {
                backgroundColor: theme.palette.primary.light,
              },
              "& .MuiDataGrid-footerContainer": {
                backgroundColor: theme.palette.background.alt,
                color: theme.palette.secondary[100],
                borderTop: "none",
              },
              "& .MuiDataGrid-toolbarContainer .MuiButton-text": {
                color: `${theme.palette.secondary[200]} !important`,
              },
            }}
        >
            <DataGrid 
                loading={isLoading || !data}
                getRowId = {(row) => row._id}
                rows={(data && data.transactions) || []}
                columns={columns}
                rowCount={(data && data.total) || 0}
                rowsPerPageOptions={[20, 50, 100]}
                pagination
                page={page}
                pageSize={pageSize}
                paginationMode="server"
                sortingMode="server"
                onPageChange={handlePageChange}
                onPageSizeChange={handlePageSizeChange}
                onSortModelChange={(newSortModel) => setSort(...newSortModel)}
                components={{ Toolbar: DataGridCustomToolbar }}
                componentsProps={{
                  toolbar: { searchInput, setSearchInput, setSearch }
                }}
            />
        </Box>
    </Box>
  )
}

export default Transactions