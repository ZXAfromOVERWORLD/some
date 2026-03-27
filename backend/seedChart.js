const mongoose = require('mongoose');
const dotenv = require('dotenv');
const Request = require('./models/Request');
const User = require('./models/User');

dotenv.config();

const seedData = async () => {
  try {
    await mongoose.connect(process.env.MONGO_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log('Connected to DB');

    const user = await User.findOne();
    if (!user) {
      console.log('No user found to associate requests to!');
      process.exit(1);
    }

    const today = new Date();
    const requestsToInsert = [];

    // Dynamic volumes to create a nice-looking organic trend chart over the last 7 days
    const volumes = [4, 9, 6, 15, 24, 11, 28]; // for 6 days ago up to today

    for (let i = 0; i < 7; i++) {
        const date = new Date(today);
        date.setDate(today.getDate() - (6 - i)); 
        date.setHours(14, 30, 0, 0);

        for (let j = 0; j < volumes[i]; j++) {
            requestsToInsert.push({
                userId: user._id,
                location: `Sector ${Math.floor(Math.random() * 50) + 1}, Block ${String.fromCharCode(65 + Math.floor(Math.random()*6))}`,
                lat: 28.6139 + (Math.random() - 0.5) * 0.1,
                lng: 77.2090 + (Math.random() - 0.5) * 0.1,
                wasteType: ['General', 'Recyclable', 'Organic', 'Hazardous', 'Electronic'][Math.floor(Math.random() * 5)],
                status: 'Completed',     // Make sure it hits the Chart completion query
                scheduledDate: date.toISOString().split('T')[0],
                createdAt: new Date(date.getTime() - 86400000 * Math.random()), // Requested day before
                updatedAt: date,         // Date completed!
            });
        }
    }

    await Request.insertMany(requestsToInsert);
    console.log(`Successfully seeded ${requestsToInsert.length} mock completed requests for the Admin Chart!`);
    process.exit(0);
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
};

seedData();
