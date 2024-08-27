import express from 'express';
import bodyParser from 'body-parser';
import mongoose, { mongo } from 'mongoose';
import cors from 'cors';
import dotenv from 'dotenv';
import helmet from 'helmet';
import morgan from 'morgan';
import clientRoutes from "./routes/client.js";
import generalRoutes from "./routes/general.js"
import managementRoutes from "./routes/management.js"
import salesRoutes from "./routes/sales.js"

// DATA IMPORTS
import User from "./models/User.js";
import Product from "./models/Product.js";
import ProductStat from "./models/ProductStat.js";
import Transaction from './models/Transaction.js';
import {
  dataUser,
  dataProduct,
  dataProductStat,
  dataTransaction,
} from "./data/index.js";

/* CONFIGURATION */
dotenv.config();
const app = express();
app.use(express.json());
app.use(helmet());
app.use(helmet.crossOriginResourcePolicy({ policy: "cross-origin" }));
app.use(morgan("common"));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cors());

/* ROUTES */
app.use("/client", clientRoutes);
app.use("/general", generalRoutes);
app.use("/management", managementRoutes);
app.use("/sales", salesRoutes);


/* MONGOOSE SETUP */
const PORT = process.env.PORT || 9000;

async function insertDataIfEmpty() {
    try {
        const collections = [
            { model: User, data: dataUser, name: 'User' },
            { model: Product, data: dataProduct, name: 'Product' },
            { model: ProductStat, data: dataProductStat, name: 'ProductStat' },
            { model: Transaction, data: dataTransaction, name: 'Transaction' }
        ];

        for (const collection of collections) {
            const count = await collection.model.countDocuments();
            if (count === 0) {
                await collection.model.insertMany(collection.data);
                console.log(`${collection.name} data inserted successfully`);
            } else {
                console.log(`PASS! ${collection.name} collection already contains data, skipping insertion`);
            }
        }

        console.log("Data insertion process completed");
    } catch (error) {
        console.error("Error during data insertion:", error);
    }
}

mongoose.connect(process.env.MONGO_URL).then(() => {
    app.listen(PORT, () => console.log(`Server Port: ${PORT}`));
    insertDataIfEmpty();
}).catch((error) => console.log(`${error} did not connect`));