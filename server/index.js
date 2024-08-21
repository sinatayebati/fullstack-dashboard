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
import { dataUser } from "./data/index.js";

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
app.use("/client", clientRoutes)
app.use("/general", generalRoutes)
app.use("/management", managementRoutes)
app.use("/sales", salesRoutes)

/* MONGOOSE SETUP */
const PORT = process.env.PORT || 9000;

async function insertDataIfEmpty() {
    const count = await User.countDocuments();
    if (count === 0) {
        try {
            await User.insertMany(dataUser);
            console.log("Sample data inserted successfully");
        } catch (error) {
            console.error("Error inserting sample data:", error);
        }
    } else {
        console.log("Database already contains data, skipping insertion");
    }
}

mongoose.connect(process.env.MONGO_URL).then(() => {
    app.listen(PORT, () => console.log(`Server Port: ${PORT}`));
    insertDataIfEmpty();
}).catch((error) => console.log(`${error} did not connect`));